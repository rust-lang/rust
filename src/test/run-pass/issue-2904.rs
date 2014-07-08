
// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]

extern crate debug;

/// Map representation

use std::io;
use std::fmt;

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

impl fmt::Show for square {
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
        println!("invalid square: {:?}", c);
        fail!()
      }
    }
}

fn read_board_grid<rdr:'static + io::Reader>(mut input: rdr)
                   -> Vec<Vec<square>> {
    let mut input: &mut io::Reader = &mut input;
    let mut grid = Vec::new();
    let mut line = [0, ..10];
    input.read(line);
    let mut row = Vec::new();
    for c in line.iter() {
        row.push(square_from_char(*c as char))
    }
    grid.push(row);
    let width = grid.get(0).len();
    for row in grid.iter() { assert!(row.len() == width) }
    grid
}

mod test {
    #[test]
    pub fn trivial_to_string() {
        assert!(lambda.to_string() == "\\")
    }
}

pub fn main() {}
