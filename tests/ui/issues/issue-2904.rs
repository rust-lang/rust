//@ build-pass
#![allow(unused_must_use)]
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(non_camel_case_types)]

// Map representation

use std::fmt;
use std::io::prelude::*;
use square::{bot, wall, rock, lambda, closed_lift, open_lift, earth, empty};

enum square {
    bot,
    wall,
    rock,
    lambda,
    closed_lift,
    open_lift,
    earth,
    empty
}

impl fmt::Debug for square {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match *self {
          bot => { "R".to_string() }
          wall => { "#".to_string() }
          rock => { "*".to_string() }
          lambda => { "\\".to_string() }
          closed_lift => { "L".to_string() }
          open_lift => { "O".to_string() }
          earth => { ".".to_string() }
          empty => { " ".to_string() }
        })
    }
}

fn square_from_char(c: char) -> square {
    match c  {
      'R'  => { bot }
      '#'  => { wall }
      '*'  => { rock }
      '\\' => { lambda }
      'L'  => { closed_lift }
      'O'  => { open_lift }
      '.'  => { earth }
      ' '  => { empty }
      _ => {
        println!("invalid square: {}", c);
        panic!()
      }
    }
}

fn read_board_grid<rdr:'static + Read>(mut input: rdr)
                   -> Vec<Vec<square>> {
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
    for row in &grid { assert_eq!(row.len(), width) }
    grid
}

mod test {
    #[test]
    pub fn trivial_to_string() {
        assert_eq!(lambda.to_string(), "\\")
    }
}

pub fn main() {}
