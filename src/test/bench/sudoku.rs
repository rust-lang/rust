// xfail-pretty

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;

use std::io::{ReaderUtil, WriterUtil};
use std::io;
use std::os;
use std::str;
use std::u8;
use std::uint;
use std::unstable::intrinsics::cttz16;
use std::vec;

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

// internal type of sudoku grids
type grid = ~[~[u8]];

struct Sudoku {
    grid: grid
}

pub impl Sudoku {
    pub fn new(g: grid) -> Sudoku {
        return Sudoku { grid: g }
    }

    pub fn from_vec(vec: &[[u8, ..9], ..9]) -> Sudoku {
        let mut g = do vec::from_fn(9u) |i| {
            do vec::from_fn(9u) |j| { vec[i][j] }
        };
        return Sudoku::new(g)
    }

    pub fn equal(&self, other: &Sudoku) -> bool {
        for u8::range(0u8, 9u8) |row| {
            for u8::range(0u8, 9u8) |col| {
                if self.grid[row][col] != other.grid[row][col] {
                    return false;
                }
            }
        }
        return true;
    }

    pub fn read(reader: @io::Reader) -> Sudoku {
        assert!(reader.read_line() == ~"9,9"); /* assert first line is exactly "9,9" */

        let mut g = vec::from_fn(10u, { |_i| ~[0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8] });
        while !reader.eof() {
            let line = reader.read_line();
            let mut comps = ~[];
            for str::each_split_char(line.trim(), ',') |s| { comps.push(s.to_owned()) }
            if comps.len() == 3u {
                let row     = uint::from_str(comps[0]).get() as u8;
                let col     = uint::from_str(comps[1]).get() as u8;
                g[row][col] = uint::from_str(comps[2]).get() as u8;
            }
            else {
                fail!("Invalid sudoku file");
            }
        }
        return Sudoku::new(g)
    }

    pub fn write(&self, writer: @io::Writer) {
        for u8::range(0u8, 9u8) |row| {
            writer.write_str(fmt!("%u", self.grid[row][0] as uint));
            for u8::range(1u8, 9u8) |col| {
                writer.write_str(fmt!(" %u", self.grid[row][col] as uint));
            }
            writer.write_char('\n');
         }
    }

    // solve sudoku grid
    pub fn solve(&mut self) {
        let mut work: ~[(u8, u8)] = ~[]; /* queue of uncolored fields */
        for u8::range(0u8, 9u8) |row| {
            for u8::range(0u8, 9u8) |col| {
                let color = self.grid[row][col];
                if color == 0u8 { work += [(row, col)]; }
            }
        }

        let mut ptr = 0u;
        let end = work.len();
        while (ptr < end) {
            let (row, col) = work[ptr];
            // is there another color to try?
            if self.next_color(row, col, self.grid[row][col] + (1 as u8)) {
                //  yes: advance work list
                ptr = ptr + 1u;
            } else {
                // no: redo this field aft recoloring pred; unless there is none
                if ptr == 0u { fail!("No solution found for this sudoku"); }
                ptr = ptr - 1u;
            }
        }
    }

    fn next_color(&mut self, row: u8, col: u8, start_color: u8) -> bool {
        if start_color < 10u8 {
            // colors not yet used
            let mut avail = ~Colors::new(start_color);

            // drop colors already in use in neighbourhood
            self.drop_colors(avail, row, col);

            // find first remaining color that is available
            let next = avail.next();
            self.grid[row][col] = next;
            return 0u8 != next;
        }
        self.grid[row][col] = 0u8;
        return false;
    }

    // find colors available in neighbourhood of (row, col)
    fn drop_colors(&mut self, avail: &mut Colors, row: u8, col: u8) {
        for u8::range(0u8, 9u8) |idx| {
            avail.remove(self.grid[idx][col]); /* check same column fields */
            avail.remove(self.grid[row][idx]); /* check same row fields */
        }

        // check same block fields
        let row0 = (row / 3u8) * 3u8;
        let col0 = (col / 3u8) * 3u8;
        for u8::range(row0, row0 + 3u8) |alt_row| {
            for u8::range(col0, col0 + 3u8) |alt_col| { avail.remove(self.grid[alt_row][alt_col]); }
        }
    }
}

// Stores available colors as simple bitfield, bit 0 is always unset
struct Colors(u16);

static heads: u16 = (1u16 << 10) - 1; /* bits 9..0 */

impl Colors {
    fn new(start_color: u8) -> Colors {
        // Sets bits 9..start_color
        let tails = !0u16 << start_color;
        return Colors(heads & tails);
    }

    fn next(&self) -> u8 {
        let val = **self & heads;
        if (0u16 == val) {
            return 0u8;
        } else {
            unsafe {
                return cttz16(val as i16) as u8;
            }
        }
    }

    fn remove(&mut self, color: u8) {
        if color != 0u8 {
            let val  = **self;
            let mask = !(1u16 << color);
            *self    = Colors(val & mask);
        }
    }
}

static default_sudoku: [[u8, ..9], ..9] = [
         /* 0    1    2    3    4    5    6    7    8    */
  /* 0 */  [0u8, 4u8, 0u8, 6u8, 0u8, 0u8, 0u8, 3u8, 2u8],
  /* 1 */  [0u8, 0u8, 8u8, 0u8, 2u8, 0u8, 0u8, 0u8, 0u8],
  /* 2 */  [7u8, 0u8, 0u8, 8u8, 0u8, 0u8, 0u8, 0u8, 0u8],
  /* 3 */  [0u8, 0u8, 0u8, 5u8, 0u8, 0u8, 0u8, 0u8, 0u8],
  /* 4 */  [0u8, 5u8, 0u8, 0u8, 0u8, 3u8, 6u8, 0u8, 0u8],
  /* 5 */  [6u8, 8u8, 0u8, 0u8, 0u8, 0u8, 0u8, 9u8, 0u8],
  /* 6 */  [0u8, 9u8, 5u8, 0u8, 0u8, 6u8, 0u8, 7u8, 0u8],
  /* 7 */  [0u8, 0u8, 0u8, 0u8, 4u8, 0u8, 0u8, 6u8, 0u8],
  /* 8 */  [4u8, 0u8, 0u8, 0u8, 0u8, 7u8, 2u8, 0u8, 3u8]
];

#[cfg(test)]
static default_solution: [[u8, ..9], ..9] = [
         /* 0    1    2    3    4    5    6    7    8    */
  /* 0 */  [1u8, 4u8, 9u8, 6u8, 7u8, 5u8, 8u8, 3u8, 2u8],
  /* 1 */  [5u8, 3u8, 8u8, 1u8, 2u8, 9u8, 7u8, 4u8, 6u8],
  /* 2 */  [7u8, 2u8, 6u8, 8u8, 3u8, 4u8, 1u8, 5u8, 9u8],
  /* 3 */  [9u8, 1u8, 4u8, 5u8, 6u8, 8u8, 3u8, 2u8, 7u8],
  /* 4 */  [2u8, 5u8, 7u8, 4u8, 9u8, 3u8, 6u8, 1u8, 8u8],
  /* 5 */  [6u8, 8u8, 3u8, 7u8, 1u8, 2u8, 5u8, 9u8, 4u8],
  /* 6 */  [3u8, 9u8, 5u8, 2u8, 8u8, 6u8, 4u8, 7u8, 1u8],
  /* 7 */  [8u8, 7u8, 2u8, 3u8, 4u8, 1u8, 9u8, 6u8, 5u8],
  /* 8 */  [4u8, 6u8, 1u8, 9u8, 5u8, 7u8, 2u8, 8u8, 3u8]
];

#[test]
fn colors_new_works() {
    assert_eq!(*Colors::new(1), 1022u16);
    assert_eq!(*Colors::new(2), 1020u16);
    assert_eq!(*Colors::new(3), 1016u16);
    assert_eq!(*Colors::new(4), 1008u16);
    assert_eq!(*Colors::new(5), 992u16);
    assert_eq!(*Colors::new(6), 960u16);
    assert_eq!(*Colors::new(7), 896u16);
    assert_eq!(*Colors::new(8), 768u16);
    assert_eq!(*Colors::new(9), 512u16);
}

#[test]
fn colors_next_works() {
    assert_eq!(Colors(0).next(), 0u8);
    assert_eq!(Colors(2).next(), 1u8);
    assert_eq!(Colors(4).next(), 2u8);
    assert_eq!(Colors(8).next(), 3u8);
    assert_eq!(Colors(16).next(), 4u8);
    assert_eq!(Colors(32).next(), 5u8);
    assert_eq!(Colors(64).next(), 6u8);
    assert_eq!(Colors(128).next(), 7u8);
    assert_eq!(Colors(256).next(), 8u8);
    assert_eq!(Colors(512).next(), 9u8);
    assert_eq!(Colors(1024).next(), 0u8);
}

#[test]
fn colors_remove_works() {
    // GIVEN
    let mut colors = Colors::new(1);

    // WHEN
    colors.remove(1);

    // THEN
    assert_eq!(colors.next(), 2u8);
}

#[test]
fn check_default_sudoku_solution() {
    // GIVEN
    let mut sudoku = Sudoku::from_vec(&default_sudoku);
    let solution   = Sudoku::from_vec(&default_solution);

    // WHEN
    sudoku.solve();

    // THEN
    assert!(sudoku.equal(&solution));
}

fn main() {
    let args        = os::args();
    let use_default = args.len() == 1u;
    let mut sudoku = if use_default {
        Sudoku::from_vec(&default_sudoku)
    } else {
        Sudoku::read(io::stdin())
    };
    sudoku.solve();
    sudoku.write(io::stdout());
}
