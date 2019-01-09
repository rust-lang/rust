// run-pass
#![allow(unused_variables)]
use foo::{x, y as fooy};
use Maybe::{Yes as MaybeYes};

pub enum Maybe { Yes, No }
mod foo {
    use super::Maybe::{self as MaybeFoo};
    pub fn x(a: MaybeFoo) {}
    pub fn y(a: i32) { println!("{}", a); }
}

pub fn main() { x(MaybeYes); fooy(10); }
