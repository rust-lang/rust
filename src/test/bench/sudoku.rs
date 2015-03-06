// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty very bad with line comments

#![feature(box_syntax, core)]
#![allow(non_snake_case)]

use std::io::prelude::*;
use std::io;
use std::iter::repeat;
use std::num::Int;
use std::env;

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
type grid = Vec<Vec<u8>>;

struct Sudoku {
    grid: grid
}

impl Sudoku {
    pub fn new(g: grid) -> Sudoku {
        return Sudoku { grid: g }
    }

    pub fn from_vec(vec: &[[u8;9];9]) -> Sudoku {
        let g = (0..9).map(|i| {
            (0..9).map(|j| { vec[i][j] }).collect()
        }).collect();
        return Sudoku::new(g)
    }

    pub fn read(reader: &mut BufRead) -> Sudoku {
        /* assert first line is exactly "9,9" */
        let mut s = String::new();
        reader.read_line(&mut s).unwrap();
        assert_eq!(s, "9,9\n");

        let mut g = repeat(vec![0, 0, 0, 0, 0, 0, 0, 0, 0])
                          .take(10).collect::<Vec<_>>();
        for line in reader.lines() {
            let line = line.unwrap();
            let comps: Vec<&str> = line
                                       .trim()
                                       .split(',')
                                       .collect();

            if comps.len() == 3 {
                let row = comps[0].parse::<u8>().unwrap();
                let col = comps[1].parse::<u8>().unwrap();
                g[row as usize][col as usize] = comps[2].parse().unwrap();
            }
            else {
                panic!("Invalid sudoku file");
            }
        }
        return Sudoku::new(g)
    }

    pub fn write(&self, writer: &mut Write) {
        for row in 0u8..9u8 {
            write!(writer, "{}", self.grid[row as usize][0]);
            for col in 1u8..9u8 {
                write!(writer, " {}", self.grid[row as usize][col as usize]);
            }
            write!(writer, "\n");
         }
    }

    // solve sudoku grid
    pub fn solve(&mut self) {
        let mut work: Vec<(u8, u8)> = Vec::new(); /* queue of uncolored fields */
        for row in 0..9 {
            for col in 0..9 {
                let color = self.grid[row as usize][col as usize];
                if color == 0 {
                    work.push((row, col));
                }
            }
        }

        let mut ptr = 0;
        let end = work.len();
        while ptr < end {
            let (row, col) = work[ptr];
            // is there another color to try?
            let the_color = self.grid[row as usize][col as usize] +
                                (1 as u8);
            if self.next_color(row, col, the_color) {
                //  yes: advance work list
                ptr = ptr + 1;
            } else {
                // no: redo this field aft recoloring pred; unless there is none
                if ptr == 0 { panic!("No solution found for this sudoku"); }
                ptr = ptr - 1;
            }
        }
    }

    fn next_color(&mut self, row: u8, col: u8, start_color: u8) -> bool {
        if start_color < 10 {
            // colors not yet used
            let mut avail: Box<_> = box Colors::new(start_color);

            // drop colors already in use in neighbourhood
            self.drop_colors(&mut *avail, row, col);

            // find first remaining color that is available
            let next = avail.next();
            self.grid[row as usize][col as usize] = next;
            return 0 != next;
        }
        self.grid[row as usize][col as usize] = 0;
        return false;
    }

    // find colors available in neighbourhood of (row, col)
    fn drop_colors(&mut self, avail: &mut Colors, row: u8, col: u8) {
        for idx in 0..9 {
            /* check same column fields */
            avail.remove(self.grid[idx as usize][col as usize]);
            /* check same row fields */
            avail.remove(self.grid[row as usize][idx as usize]);
        }

        // check same block fields
        let row0 = (row / 3) * 3;
        let col0 = (col / 3) * 3;
        for alt_row in row0..row0 + 3 {
            for alt_col in col0..col0 + 3 {
                avail.remove(self.grid[alt_row as usize][alt_col as usize]);
            }
        }
    }
}

// Stores available colors as simple bitfield, bit 0 is always unset
struct Colors(u16);

static HEADS: u16 = (1 << 10) - 1; /* bits 9..0 */

impl Colors {
    fn new(start_color: u8) -> Colors {
        // Sets bits 9..start_color
        let tails = !0 << start_color as usize;
        return Colors(HEADS & tails);
    }

    fn next(&self) -> u8 {
        let Colors(c) = *self;
        let val = c & HEADS;
        if 0 == val {
            return 0;
        } else {
            return val.trailing_zeros() as u8
        }
    }

    fn remove(&mut self, color: u8) {
        if color != 0 {
            let Colors(val) = *self;
            let mask = !(1 << color as usize);
            *self    = Colors(val & mask);
        }
    }
}

static DEFAULT_SUDOKU: [[u8;9];9] = [
         /* 0    1    2    3    4    5    6    7    8    */
  /* 0 */  [0, 4, 0, 6, 0, 0, 0, 3, 2],
  /* 1 */  [0, 0, 8, 0, 2, 0, 0, 0, 0],
  /* 2 */  [7, 0, 0, 8, 0, 0, 0, 0, 0],
  /* 3 */  [0, 0, 0, 5, 0, 0, 0, 0, 0],
  /* 4 */  [0, 5, 0, 0, 0, 3, 6, 0, 0],
  /* 5 */  [6, 8, 0, 0, 0, 0, 0, 9, 0],
  /* 6 */  [0, 9, 5, 0, 0, 6, 0, 7, 0],
  /* 7 */  [0, 0, 0, 0, 4, 0, 0, 6, 0],
  /* 8 */  [4, 0, 0, 0, 0, 7, 2, 0, 3]
];

#[cfg(test)]
static DEFAULT_SOLUTION: [[u8;9];9] = [
         /* 0    1    2    3    4    5    6    7    8    */
  /* 0 */  [1, 4, 9, 6, 7, 5, 8, 3, 2],
  /* 1 */  [5, 3, 8, 1, 2, 9, 7, 4, 6],
  /* 2 */  [7, 2, 6, 8, 3, 4, 1, 5, 9],
  /* 3 */  [9, 1, 4, 5, 6, 8, 3, 2, 7],
  /* 4 */  [2, 5, 7, 4, 9, 3, 6, 1, 8],
  /* 5 */  [6, 8, 3, 7, 1, 2, 5, 9, 4],
  /* 6 */  [3, 9, 5, 2, 8, 6, 4, 7, 1],
  /* 7 */  [8, 7, 2, 3, 4, 1, 9, 6, 5],
  /* 8 */  [4, 6, 1, 9, 5, 7, 2, 8, 3]
];

#[test]
fn colors_new_works() {
    assert_eq!(*Colors::new(1), 1022);
    assert_eq!(*Colors::new(2), 1020);
    assert_eq!(*Colors::new(3), 1016);
    assert_eq!(*Colors::new(4), 1008);
    assert_eq!(*Colors::new(5), 992);
    assert_eq!(*Colors::new(6), 960);
    assert_eq!(*Colors::new(7), 896);
    assert_eq!(*Colors::new(8), 768);
    assert_eq!(*Colors::new(9), 512);
}

#[test]
fn colors_next_works() {
    assert_eq!(Colors(0).next(), 0);
    assert_eq!(Colors(2).next(), 1);
    assert_eq!(Colors(4).next(), 2);
    assert_eq!(Colors(8).next(), 3);
    assert_eq!(Colors(16).next(), 4);
    assert_eq!(Colors(32).next(), 5);
    assert_eq!(Colors(64).next(), 6);
    assert_eq!(Colors(128).next(), 7);
    assert_eq!(Colors(256).next(), 8);
    assert_eq!(Colors(512).next(), 9);
    assert_eq!(Colors(1024).next(), 0);
}

#[test]
fn colors_remove_works() {
    // GIVEN
    let mut colors = Colors::new(1);

    // WHEN
    colors.remove(1);

    // THEN
    assert_eq!(colors.next(), 2);
}

#[test]
fn check_DEFAULT_SUDOKU_solution() {
    // GIVEN
    let mut sudoku = Sudoku::from_vec(&DEFAULT_SUDOKU);
    let solution   = Sudoku::from_vec(&DEFAULT_SOLUTION);

    // WHEN
    sudoku.solve();

    // THEN
    assert!(sudoku.equal(&solution));
}

fn main() {
    let args = env::args();
    let use_default = args.len() == 1;
    let mut sudoku = if use_default {
        Sudoku::from_vec(&DEFAULT_SUDOKU)
    } else {
        let stdin = io::stdin();
        let mut locked = stdin.lock();
        Sudoku::read(&mut locked)
    };
    sudoku.solve();
    let out = io::stdout();
    sudoku.write(&mut out.lock());
}
