// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// Testing creating two vtables with the same self type, but different
// traits.

#![feature(box_syntax)]

use std::any::Any;

trait Wrap {
    fn get(&self) -> isize;
    fn wrap(self: Box<Self>) -> Box<dyn Any+'static>;
}

impl Wrap for isize {
    fn get(&self) -> isize {
        *self
    }
    fn wrap(self: Box<isize>) -> Box<dyn Any+'static> {
        self as Box<dyn Any+'static>
    }
}

fn is<T:Any>(x: &dyn Any) -> bool {
    x.is::<T>()
}

fn main() {
    let x = box 22isize as Box<dyn Wrap>;
    println!("x={}", x.get());
    let y = x.wrap();
}
