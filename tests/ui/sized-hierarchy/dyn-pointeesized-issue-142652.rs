#![feature(sized_hierarchy)]

use std::marker::PointeeSized;

fn main() {
      let x = main;
      let y: Box<dyn PointeeSized> = x;
//~^ ERROR mismatched types
}
