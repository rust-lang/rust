// run-pass
#![allow(unused_imports)]

#![allow(dead_code)]

// this code used to cause an ICE

use std::marker;

trait X<T> {
    fn dummy(&self) -> T { panic!() }
}

struct S<T> {f: Box<dyn X<T>+'static>,
             g: Box<dyn X<T>+'static>}

struct F;
impl X<isize> for F {
}

fn main() {
  S {f: Box::new(F), g: Box::new(F) };
}
