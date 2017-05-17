#![feature(plugin)]
#![plugin(clippy)]
#![warn(clippy)]
#![allow(unused, needless_pass_by_value)]
#![feature(associated_consts, associated_type_defaults)]

type Alias = Vec<Vec<Box<(u32, u32, u32, u32)>>>; // no warning here

const CST: (u32, (u32, (u32, (u32, u32)))) = (0, (0, (0, (0, 0))));
static ST: (u32, (u32, (u32, (u32, u32)))) = (0, (0, (0, (0, 0))));

struct S {
    f: Vec<Vec<Box<(u32, u32, u32, u32)>>>,
}

struct TS(Vec<Vec<Box<(u32, u32, u32, u32)>>>);

enum E {
    Tuple(Vec<Vec<Box<(u32, u32, u32, u32)>>>),
    Struct { f: Vec<Vec<Box<(u32, u32, u32, u32)>>> },
}

impl S {
    const A: (u32, (u32, (u32, (u32, u32)))) = (0, (0, (0, (0, 0))));
    fn impl_method(&self, p: Vec<Vec<Box<(u32, u32, u32, u32)>>>) { }
}

trait T {
    const A: Vec<Vec<Box<(u32, u32, u32, u32)>>>;
    type B = Vec<Vec<Box<(u32, u32, u32, u32)>>>;
    fn method(&self, p: Vec<Vec<Box<(u32, u32, u32, u32)>>>);
    fn def_method(&self, p: Vec<Vec<Box<(u32, u32, u32, u32)>>>) { }
}

fn test1() -> Vec<Vec<Box<(u32, u32, u32, u32)>>> { vec![] }

fn test2(_x: Vec<Vec<Box<(u32, u32, u32, u32)>>>) { }

fn test3() {
    let _y: Vec<Vec<Box<(u32, u32, u32, u32)>>> = vec![];
}

fn main() {
}
