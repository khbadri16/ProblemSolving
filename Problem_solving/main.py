from __future__ import annotations
import copy
import hashlib
import heapq
from collections import deque
import random
from representation2 import GridVisualizer

class Grid:
    def __init__(self, board: list[list[int]]):
        """This is the constructor of a puzzle. It takes a matrix to initialize the board

        Args:
            board (list[list[int]]): the initial board
        """
        self.board = copy.deepcopy(board)
        self.parent = None
        self.row = None
        self.col = None
        self.depth = 0
        hash(self)

    def set_row_col(self, row: int, col: int):
        """This methods sets the starting point

        Args:
            row (int): row of the starting point
            col (int): column of the starting point
        """
        self.row = row
        self.col = col
        self.board[row][col] = 2

    def __stringify_grid(grid) -> str:
        """This is a utility function to produce a pretty representation of the board

        Args:
            grid (_type_): a board

        Returns:
            str: a pretty representation of the grid
        """
        chars = " ~#"
        return "\n".join(["".join([f"[{chars[c]}]" for c in row]) for row in grid])

    def __str__(self) -> str:
        """Utility function to have a string representation of the puzzle

        Returns:
            str: the string representation
        """
        return Grid.__stringify_grid(self.board)

    def __hash__(self) -> int:
        """Computes the hash code of the puzzle in order to accelerate puzzle comparison. This methods uses
        the MD5 function to compute a hash code.

        Returns:
            int: the hash code
        """
        s = "".join(["".join([f"{c}" for c in row]) for row in self.board])
        s+=str(self.row)+","+str(self.col)
        self.__hash_code = int.from_bytes(hashlib.md5(s.encode()).digest(), "little")
        return self.__hash_code

    def __eq__(self, value: Grid) -> bool:
        """Tests whether two grids are equal.

        Args:
            value (Grid): the grid to be compared to

        Returns:
            bool: the result of the comparison
        """
        return self.__hash_code == value.__hash_code

    def clone(self, row: int, col: int) -> Grid:
        """Creates a copy of the current puzzle and set the current position to (row,col)

        Args:
            row (int): the row of the new position
            col (int): the column of the new position

        Returns:
            Grid: _description_
        """
        v = Grid(self.board)
        v.set_row_col(row, col)
        hash(v)
        v.parent = self
        v.depth = self.depth + 1
        return v

    def is_goal(self) -> bool:
        """Tests if the current puzzle is the goal (it does not contains 1)

        Returns:
            bool: True if if is the goal, False otherwise
        """
        for row in self.board:
            if 1 in row:
                return False
        return True

    def actions(self) -> list[Grid]:
        """This function generates possible actions from the current puzzle. Each element in the return list is new puzzle

        Returns:
            list[Grid]: the list of the new puzzles
        """
        tries = []
        if self.row > 0 and self.board[self.row - 1][self.col] == 1:
            tries.append((self.row - 1, self.col))
        if self.row < len(self.board) - 1 and self.board[self.row + 1][self.col] == 1:
            tries.append((self.row + 1, self.col))
        if self.col > 0 and self.board[self.row][self.col - 1] == 1:
            tries.append((self.row, self.col - 1))
        if self.col < len(self.board[0]) - 1 and self.board[self.row][self.col + 1] == 1:
            tries.append((self.row, self.col + 1))

        res = []
        for row1, col1 in tries:
            v = self.clone(row1, col1)
            res.append(v)
        return res

    def solve_breadth(self):
        """Solve the problem with a Breadth First Algorithm."""
        ouvert = deque([self])
        state_count = 0
        ferme = set()

        while ouvert:
            noeud = ouvert.popleft()
            ferme.add(noeud)
            state_count += 1
            print(f"Exploring state {state_count}, depth : {noeud.depth}")

            if noeud.is_goal():
                path_solution = []
                while noeud:
                    path_solution.append(noeud)
                    noeud = noeud.parent
                return path_solution[::-1]

            for m in noeud.actions():
                if m not in ferme :
                    ouvert.append(m)
        return []


    def solve_depth(self):
        """Solve the problem with a Depth First Algorithm."""
        ouvert = [self]
        state_count = 0
        ferme = set()

        while ouvert:
            noeud = ouvert.pop()
            ferme.add(noeud)
            state_count += 1

            print(f"Exploring state {state_count}, depth : {noeud.depth}")

            if noeud.is_goal():
                path_solution = []
                while noeud:
                    path_solution.append(noeud)
                    noeud = noeud.parent
                return path_solution[::-1]

            for m in noeud.actions():
                if m not in ferme:
                    ouvert.append(m)
        return []

    def solve_random(self):
            """Solve the problem with a Random Algorithm (based on the Open/Closed algorithm)"""
            ouvert_stack = [self]
            ferme_lookup = set()
            state_count = 0

            while ouvert_stack:
                idx = random.randrange(len(ouvert_stack))
                ouvert_stack[idx], ouvert_stack[-1] = ouvert_stack[-1], ouvert_stack[idx]
                noeud = ouvert_stack.pop()
                ferme_lookup.add(noeud)
                state_count += 1
                print(f"Exploring state {state_count}, depth : {noeud.depth}")

                if noeud.is_goal():
                    path_solution = []
                    while noeud:
                        path_solution.append(noeud)
                        noeud = noeud.parent
                    return path_solution[::-1]

                for m in noeud.actions():
                    if m not in ferme_lookup:
                        ouvert_stack.append(m)
            return []


    def solve_heur(self,a,b,c):
        """Solve the problem with a Heuristic Algorithm according to what has been explained in the statements"""
        ouvert_stack = []
        heapq.heappush(ouvert_stack, (self.evaluation_func(position=(self.row,self.col),a=a,b=b,c=c), id(self), self))
        ferme_lookup = set()
        state_count = 0

        while ouvert_stack:
            eval_score, _ , noeud = heapq.heappop(ouvert_stack)
            ferme_lookup.add(noeud)
            state_count += 1
            print(f"{state_count=}, Depth: {noeud.depth}, eval={eval_score}")

            if noeud.is_goal():
                path_solution = []
                while noeud:
                    path_solution.append(noeud)
                    noeud = noeud.parent
                return path_solution[::-1]

            for m in noeud.actions():
                if m not in ferme_lookup :
                    heapq.heappush(ouvert_stack, (m.evaluation_func(position=(m.row, m.col), a=a, b=b, c=c), id(m), m,))

        return []

    def evaluation_func(self,position,a,b,c):
        world_3,row_3,col_3 = self.get_the_world(3,position=position)
        world_5,row_5,col_5= self.get_the_world(5,position=position)
        number_of_ones_in_k3_world = sum(row.count(1) for row in world_3)
        number_of_ones_in_k5_world = sum(row.count(1) for row in world_5)
        p_3 = number_of_ones_in_k3_world - solve_rec(world_3,
                                                     x=row_3,
                                                     y=col_3,
                                                     nb=0,
                                                     n=number_of_ones_in_k3_world)
        p_5 = number_of_ones_in_k5_world - solve_rec(world_5,
                                                     x=row_5,
                                                     y=col_5,
                                                     nb=0,
                                                     n=number_of_ones_in_k5_world)
        return  (a*p_5) + (b*p_3) - (self.depth/c)


    def get_the_world(self, k: int, position: tuple[int, int]):
        row, col = position
        half_k = k // 2
        start_row = max(0, row - half_k)
        end_row = min(len(self.board), row + half_k + 1)
        start_col = max(0, col - half_k)
        end_col = min(len(self.board[0]), col + half_k + 1)

        world = [r[start_col:end_col] for r in self.board[start_row:end_row]]

        relative_row = row - start_row
        relative_col = col - start_col

        return copy.deepcopy(world) , relative_row , relative_col


def solve_rec(world, x: int, y: int, nb: int, n: int) -> int:
    """This function solves the problem with a backtracking recursive algorithm. It should be used to compute
    the values p_k (as explained in the statements). p_k=n-solve_rec(world,x,y,0,n), such as n is the initial number
    of 1s in the world variable.
    <br>
    To use this function, you should prepare a k*k matrix (called world here) around the current position (x,y) from the puzzle (k=3 or 5). You should
    count the number of 1s in the world (let it be n) then compute the value n-solve_rec(world,x,y,0,n).

    Args:
        world (_type_): a matrix of size k*k to be built
        x (int): the row of the current position
        y (int): the column of the current position
        nb (int): initial number of green cells (should be 0 at the first call)
        n (int): the total number of 1s in the world (the k*k matrix)

    Returns:
        int: the maximum number of blue cells that can be covered
    """
    if nb == n:
        return n
    tries = []
    if x > 0 and world[x - 1][y] == 1:
        tries.append((x - 1, y))
    if x < len(world) - 1 and world[x + 1][y] == 1:
        tries.append((x + 1, y))
    if y > 0 and world[x][y - 1] == 1:
        tries.append((x, y - 1))
    if y < len(world[0]) - 1 and world[x][y + 1] == 1:
        tries.append((x, y + 1))
    mx = nb
    for x1, y1 in tries:
        world[x1][y1] = 2
        sol = solve_rec(world, x1, y1, nb + 1, n)
        world[x1][y1] = 1
        if mx < sol:
            mx = sol
        if mx == n:
            return n

    return mx


if __name__ == "__main__":
    # 1 ,0, 50
    benchmark1 = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
    # 0 ,1 ,50
    benchmark2 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]]
    # 1, 1, 50
    benchmark3 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                  [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
    # 1, 1, 50
    benchmark4 = [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
                  [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # 1, 1, 50
    benchmark5 = [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
                  [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0],
                  [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]
    # any
    benchmark7 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]
    # any
    benchmark8 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]



    gr = Grid(benchmark1)
    # for all benchmarks, the starting point is (3,3)
    gr.set_row_col(3, 3)
    path = gr.solve_breadth()
    # path = gr.solve_depth()
    # path = gr.solve_random()
    # path = gr.solve_heur(1, 0, 50)  #benchmark1
    # path = gr.solve_heur(0, 1, 50)  #benchmark2
    # path = gr.solve_heur(1, 1, 50) #benchmark3 , benchmark4 , #benchmark5
    if path:
      GridVisualizer(path)
    else:
        print('No solution')


