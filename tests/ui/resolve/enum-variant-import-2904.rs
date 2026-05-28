//! Regression test for https://github.com/rust-lang/rust/issues/2904

//@ build-pass
#![allow(unused_must_use)]
#![allow(dead_code)]
#![allow(unused_mut)]

// Map representation

use Square::{Bot, ClosedLift, Earth, Empty, Lambda, OpenLift, Rock, Wall};
use std::fmt;
use std::io::prelude::*;

enum Square {
    Bot,
    Wall,
    Rock,
    Lambda,
    ClosedLift,
    OpenLift,
    Earth,
    Empty,
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match *self {
                Bot => {
                    "R".to_string()
                }
                Wall => {
                    "#".to_string()
                }
                Rock => {
                    "*".to_string()
                }
                Lambda => {
                    "\\".to_string()
                }
                ClosedLift => {
                    "L".to_string()
                }
                OpenLift => {
                    "O".to_string()
                }
                Earth => {
                    ".".to_string()
                }
                Empty => {
                    " ".to_string()
                }
            }
        )
    }
}

fn square_from_char(c: char) -> Square {
    match c {
        'R' => Bot,
        '#' => Wall,
        '*' => Rock,
        '\\' => Lambda,
        'L' => ClosedLift,
        'O' => OpenLift,
        '.' => Earth,
        ' ' => Empty,
        _ => {
            println!("invalid Square: {}", c);
            panic!()
        }
    }
}

fn read_board_grid<Rdr: 'static + Read>(mut input: Rdr) -> Vec<Vec<Square>> {
    let mut input: &mut dyn Read = &mut input;
    let mut grid = Vec::new();
    let mut line = [0; 10];
    input.read(&mut line);
    let mut row = Vec::new();
    for c in &line {
        row.push(square_from_char(*c as char))
    }
    grid.push(row);
    let width = grid[0].len();
    for row in &grid {
        assert_eq!(row.len(), width)
    }
    grid
}

mod test {
    #[test]
    pub fn trivial_to_string() {
        assert_eq!(Lambda.to_string(), "\\")
    }
}

pub fn main() {}
