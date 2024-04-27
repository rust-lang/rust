//@ run-pass

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

// Example from lkuper's intern talk, August 2012.
use Color::{cyan, magenta, yellow, black};
use ColorTree::{leaf, branch};

trait Equal {
    fn isEq(&self, a: &Self) -> bool;
}

#[derive(Clone, Copy)]
enum Color { cyan, magenta, yellow, black }

impl Equal for Color {
    fn isEq(&self, a: &Color) -> bool {
        match (*self, *a) {
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
    fn isEq(&self, a: &ColorTree) -> bool {
        match (self, a) {
          (&leaf(ref x), &leaf(ref y)) => { x.isEq(&(*y).clone()) }
          (&branch(ref l1, ref r1), &branch(ref l2, ref r2)) => {
            (*l1).isEq(&(**l2).clone()) && (*r1).isEq(&(**r2).clone())
          }
          _ => { false }
        }
    }
}

pub fn main() {
    assert!(cyan.isEq(&cyan));
    assert!(magenta.isEq(&magenta));
    assert!(!cyan.isEq(&yellow));
    assert!(!magenta.isEq(&cyan));

    assert!(leaf(cyan).isEq(&leaf(cyan)));
    assert!(!leaf(cyan).isEq(&leaf(yellow)));

    assert!(branch(Box::new(leaf(magenta)), Box::new(leaf(cyan)))
        .isEq(&branch(Box::new(leaf(magenta)), Box::new(leaf(cyan)))));

    assert!(!branch(Box::new(leaf(magenta)), Box::new(leaf(cyan)))
        .isEq(&branch(Box::new(leaf(magenta)), Box::new(leaf(magenta)))));

    println!("Assertions all succeeded!");
}
