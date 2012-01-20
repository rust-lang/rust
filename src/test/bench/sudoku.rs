use std;

import std::{io, bitv};
import io::{writer_util, reader_util};

// Computes a single solution to a given 9x9 sudoku
//
// Call with "-" to read input sudoku from stdin
//
// The expected line-based format is:
//
// 9,9
// <row>,<column>,<color>
// ...
//
// Row and column are 0-based (i.e. <= 8) and color is 1-based (>=1,<=9).
// A color of 0 indicates an empty field.
//
// If called without arguments, sudoku solves a built-in example sudoku
//

export grid_t, read_grid, solve_grid, write_grid;

// internal type of sudoku grids
type grid = [[mutable u8]];

// exported type of sudoku grids
enum grid_t { grid_ctor(grid), }

// read a sudoku problem from file f
fn read_grid(f: io::reader) -> grid_t {
    assert f.read_line() == "9,9"; /* assert first line is exactly "9,9" */

    let g = vec::init_fn({|_i| vec::init_elt_mut(0 as u8, 10u) }, 10u);
    while !f.eof() {
        // FIXME: replace with unicode compliant call
        let comps = str::split(str::trim(f.read_line()), ',' as u8);
        if vec::len(comps) >= 3u {
            let row     = uint::from_str(comps[0]) as u8;
            let col     = uint::from_str(comps[1]) as u8;
            g[row][col] = uint::from_str(comps[2]) as u8;
        }
    }
    ret grid_ctor(g);
}

// solve sudoku grid
fn solve_grid(g: grid_t) {
    fn next_color(g: grid, row: u8, col: u8, start_color: u8) -> bool {
        if start_color < 10u8 {
            // colors not yet used
            let avail = bitv::create(10u, false);
            u8::range(start_color, 10u8) { |color|
                bitv::set(avail, color as uint, true);
            }

            // drop colors already in use in neighbourhood
            drop_colors(g, avail, row, col);

            // find first remaining color that is available
            let i = 1 as uint;
            while i < (10 as uint) { /* FIXME llvm ctlhd */
                if bitv::get(avail, i) {
                    g[row][col] = i as u8;
                    ret true;
                }
                i += 1 as uint; /* else */
            }
        }
        g[row][col] = 0u8;
        ret false;
    }

    // find colors available in neighbourhood of (row, col)
    fn drop_colors(g: grid, avail: bitv::t, row: u8, col: u8) {
        fn drop_color(g: grid, colors: bitv::t, row: u8, col: u8) {
            let color = g[row][col];
            if color != 0u8 { bitv::set(colors, color as uint, false); }
        }

        let it = bind drop_color(g, avail, _, _);

        u8::range(0u8, 9u8) { |idx|
            it(idx, col); /* check same column fields */
            it(row, idx); /* check same row fields */
        }

        // check same block fields
        let row0 = (row / 3u8) * 3u8;
        let col0 = (col / 3u8) * 3u8;
        u8::range(row0, row0 + 3u8) { |alt_row|
            u8::range(col0, col0 + 3u8) { |alt_col| it(alt_row, alt_col); }
        }
    }

    let work: [(u8, u8)] = []; /* queue of uncolored fields */
    u8::range(0u8, 9u8) { |row|
        u8::range(0u8, 9u8) { |col|
            let color = (*g)[row][col];
            if color == 0u8 { work += [(row, col)]; }
        }
    }

    let ptr = 0u;
    let end = vec::len(work);
    while (ptr < end) {
        let (row, col) = work[ptr];
        // is there another color to try?
        if next_color(*g, row, col, (*g)[row][col] + (1 as u8)) {
            //  yes: advance work list
            ptr = ptr + 1u;
        } else {
            // no: redo this field aft recoloring pred; unless there is none
            if ptr == 0u { fail "No solution found for this sudoku"; }
            ptr = ptr - 1u;
        }
    }
}

fn write_grid(f: io::writer, g: grid_t) {
    u8::range(0u8, 9u8) { |row|
        f.write_str(#fmt("%u", (*g)[row][0] as uint));
        u8::range(1u8, 9u8) { |col|
            f.write_str(#fmt(" %u", (*g)[row][col] as uint));
        }
        f.write_char('\n');
     }
}

fn main(args: [str]) {
    let grid = if vec::len(args) == 1u {
        // FIXME create sudoku inline since nested vec consts dont work yet
        let g = vec::init_fn({|_i| vec::init_elt_mut(0 as u8, 10u) }, 10u);
        g[0][1] = 4u8;
        g[0][3] = 6u8;
        g[0][7] = 3u8;
        g[0][8] = 2u8;
        g[1][2] = 8u8;
        g[1][4] = 2u8;
        g[2][0] = 7u8;
        g[2][3] = 8u8;
        g[3][3] = 5u8;
        g[4][1] = 5u8;
        g[4][5] = 3u8;
        g[4][6] = 6u8;
        g[5][0] = 6u8;
        g[5][1] = 8u8;
        g[5][7] = 9u8;
        g[6][1] = 9u8;
        g[6][2] = 5u8;
        g[6][5] = 6u8;
        g[6][7] = 7u8;
        g[7][4] = 4u8;
        g[7][7] = 6u8;
        g[8][0] = 4u8;
        g[8][5] = 7u8;
        g[8][6] = 2u8;
        g[8][8] = 3u8;
        grid_ctor(g)
    } else {
        read_grid(io::stdin())
    };
    solve_grid(grid);
    write_grid(io::stdout(), grid);
}

