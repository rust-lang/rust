//@ run-pass

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

// Example from lkuper's intern talk, August 2012 -- now with static
// methods!
use Color::{cyan, magenta, yellow, black};
use ColorTree::{leaf, branch};

trait Equal {
    fn isEq(a: &Self, b: &Self) -> bool;
}

#[derive(Clone, Copy)]
enum Color { cyan, magenta, yellow, black }

impl Equal for Color {
    fn isEq(a: &Color, b: &Color) -> bool {
        match (*a, *b) {
          (cyan, cyan)       => { true  }
          (magenta, magenta) => { true  }
          (yellow, yellow)   => { true  }
          (black, black)     => { true  }
          _                  => { false }
        }
    }
}

#[derive(Clone)]
enum ColorTree {
    leaf(Color),
    branch(Box<ColorTree>, Box<ColorTree>)
}

impl Equal for ColorTree {
    fn isEq(a: &ColorTree, b: &ColorTree) -> bool {
        match (a, b) {
          (&leaf(ref x), &leaf(ref y)) => {
              Equal::isEq(&(*x).clone(), &(*y).clone())
          }
          (&branch(ref l1, ref r1), &branch(ref l2, ref r2)) => {
            Equal::isEq(&(**l1).clone(), &(**l2).clone()) &&
                Equal::isEq(&(**r1).clone(), &(**r2).clone())
          }
          _ => { false }
        }
    }
}

pub fn main() {
    assert!(Equal::isEq(&cyan, &cyan));
    assert!(Equal::isEq(&magenta, &magenta));
    assert!(!Equal::isEq(&cyan, &yellow));
    assert!(!Equal::isEq(&magenta, &cyan));

    assert!(Equal::isEq(&leaf(cyan), &leaf(cyan)));
    assert!(!Equal::isEq(&leaf(cyan), &leaf(yellow)));

    assert!(Equal::isEq(&branch(Box::new(leaf(magenta)), Box::new(leaf(cyan))),
                &branch(Box::new(leaf(magenta)), Box::new(leaf(cyan)))));

    assert!(!Equal::isEq(&branch(Box::new(leaf(magenta)), Box::new(leaf(cyan))),
                 &branch(Box::new(leaf(magenta)), Box::new(leaf(magenta)))));

    println!("Assertions all succeeded!");
}
