//@ run-pass
#![allow(dead_code)]
//@ aux-build:issue-4208-cc.rs


extern crate numeric;
use numeric::{sin, Angle};

fn foo<T, A:Angle<T>>(theta: A) -> T { sin(&theta) }

pub fn main() {}
