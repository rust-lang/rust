// run-rustfix
#![warn(clippy::all)]
#![allow(clippy::boxed_local, clippy::needless_pass_by_value)]
#![allow(clippy::blacklisted_name, unused_variables, dead_code)]

use std::boxed::Box;
use std::rc::Rc;

pub struct MyStruct {}

pub struct SubT<T> {
    foo: T,
}

pub enum MyEnum {
    One,
    Two,
}

// Rc<&T>

pub fn test1<T>(foo: Rc<&T>) {}

pub fn test2(foo: Rc<&MyStruct>) {}

pub fn test3(foo: Rc<&MyEnum>) {}

pub fn test4_neg(foo: Rc<SubT<&usize>>) {}

// Rc<Rc<T>>

pub fn test5(a: Rc<Rc<bool>>) {}

// Rc<Box<T>>

pub fn test6(a: Rc<Box<bool>>) {}

// Box<&T>

pub fn test7<T>(foo: Box<&T>) {}

pub fn test8(foo: Box<&MyStruct>) {}

pub fn test9(foo: Box<&MyEnum>) {}

pub fn test10_neg(foo: Box<SubT<&usize>>) {}

fn main() {}
