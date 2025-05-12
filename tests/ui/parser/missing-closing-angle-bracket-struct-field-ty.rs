// run-rustifx
#![allow(unused)]
use std::sync::{Arc, Mutex};

pub struct Foo {
    a: Mutex<usize>,
    b: Arc<Mutex<usize>, //~ HELP you might have meant to end the type parameters here
    c: Arc<Mutex<usize>>,
} //~ ERROR expected one of

fn main() {}
