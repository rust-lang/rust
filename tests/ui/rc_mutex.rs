#![warn(clippy::rc_mutex)]
#![allow(unused_imports)]
#![allow(clippy::boxed_local, clippy::needless_pass_by_value)]
#![allow(clippy::blacklisted_name, unused_variables, dead_code)]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Mutex;

pub struct MyStruct {}

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
