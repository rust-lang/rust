#![feature(plugin)]
#![plugin(clippy)]
#![deny(clippy)]
#![allow(unused)]
#![feature(associated_consts, associated_type_defaults)]

type Alias = Vec<Vec<Box<(u32, u32, u32, u32)>>>; // no warning here

const CST: (u32, (u32, (u32, (u32, u32)))) = (0, (0, (0, (0, 0)))); //~ERROR very complex type
static ST: (u32, (u32, (u32, (u32, u32)))) = (0, (0, (0, (0, 0)))); //~ERROR very complex type

struct S {
    f: Vec<Vec<Box<(u32, u32, u32, u32)>>>, //~ERROR very complex type
}

struct TS(Vec<Vec<Box<(u32, u32, u32, u32)>>>); //~ERROR very complex type

enum E {
    V1(Vec<Vec<Box<(u32, u32, u32, u32)>>>), //~ERROR very complex type
    V2 { f: Vec<Vec<Box<(u32, u32, u32, u32)>>> }, //~ERROR very complex type
}

impl S {
    const A: (u32, (u32, (u32, (u32, u32)))) = (0, (0, (0, (0, 0)))); //~ERROR very complex type
    fn impl_method(&self, p: Vec<Vec<Box<(u32, u32, u32, u32)>>>) { } //~ERROR very complex type
}

trait T {
    const A: Vec<Vec<Box<(u32, u32, u32, u32)>>>; //~ERROR very complex type
    type B = Vec<Vec<Box<(u32, u32, u32, u32)>>>; //~ERROR very complex type
    fn method(&self, p: Vec<Vec<Box<(u32, u32, u32, u32)>>>); //~ERROR very complex type
    fn def_method(&self, p: Vec<Vec<Box<(u32, u32, u32, u32)>>>) { } //~ERROR very complex type
}

fn test1() -> Vec<Vec<Box<(u32, u32, u32, u32)>>> { vec![] } //~ERROR very complex type

fn test2(_x: Vec<Vec<Box<(u32, u32, u32, u32)>>>) { } //~ERROR very complex type

fn test3() {
    let _y: Vec<Vec<Box<(u32, u32, u32, u32)>>> = vec![]; //~ERROR very complex type
}

fn main() {
}
