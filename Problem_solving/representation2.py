import tkinter as tk
import time


class GridVisualizer:
    def __init__(self, solution_path, cell_size=40):
        self.solution_path = solution_path
        self.cell_size = cell_size
        self.rows = len(solution_path[0].board)
        self.cols = len(solution_path[0].board[0])

        self.root = tk.Tk()
        self.root.title("Pathfinding Solution")

        self.canvas = tk.Canvas(self.root, width=self.cols * cell_size, height=self.rows * cell_size)
        self.canvas.pack()

        self.animate_solution()

        self.root.mainloop()

    def draw_grid(self, grid):
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                color = "black" if grid.board[r][c] == 0 else "blue"
                if grid.board[r][c] == 2:
                    color = "green"  # Path
                self.canvas.create_rectangle(
                    c * self.cell_size, r * self.cell_size,
                    (c + 1) * self.cell_size, (r + 1) * self.cell_size,
                    fill=color, outline="white"
                )

    def animate_solution(self):
        for grid in self.solution_path:
            self.draw_grid(grid)
            self.root.update()
            time.sleep(0.3)
