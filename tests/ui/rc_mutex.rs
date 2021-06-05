#![warn(clippy::rc_mutex)]
#![allow(clippy::boxed_local, clippy::needless_pass_by_value)]
#![allow(clippy::blacklisted_name, unused_variables, dead_code)]

use std::rc::Rc;
use std::sync::Mutex;

pub struct MyStruct {
    foo: Rc<Mutex<i32>>,
}

pub struct SubT<T> {
    foo: T,
}

pub enum MyEnum {
    One,
    Two,
}

pub fn test1<T>(foo: Rc<Mutex<T>>) {}

pub fn test2(foo: Rc<Mutex<MyEnum>>) {}

pub fn test3(foo: Rc<Mutex<SubT<usize>>>) {}

fn main() {}
