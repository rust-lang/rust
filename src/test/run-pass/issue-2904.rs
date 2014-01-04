// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

/// Map representation

extern mod extra;

use std::io;
use std::to_str;

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

impl to_str::ToStr for square {
    fn to_str(&self) -> ~str {
        match *self {
          bot => { ~"R" }
          wall => { ~"#" }
          rock => { ~"*" }
          lambda => { ~"\\" }
          closed_lift => { ~"L" }
          open_lift => { ~"O" }
          earth => { ~"." }
          empty => { ~" " }
        }
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
        error!("invalid square: {:?}", c);
        fail!()
      }
    }
}

fn read_board_grid<rdr:'static + io::Reader>(mut input: rdr) -> ~[~[square]] {
    let mut input: &mut io::Reader = &mut input;
    let mut grid = ~[];
    let mut line = [0, ..10];
    input.read(line);
    let mut row = ~[];
    for c in line.iter() {
        row.push(square_from_char(*c as char))
    }
    grid.push(row);
    let width = grid[0].len();
    for row in grid.iter() { assert!(row.len() == width) }
    grid
}

mod test {
    #[test]
    pub fn trivial_to_str() {
        assert!(lambda.to_str() == "\\")
    }
}

pub fn main() {}
