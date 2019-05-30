#![deny(clippy::borrowed_box)]
#![allow(clippy::blacklisted_name)]
#![allow(unused_variables)]
#![allow(dead_code)]

pub fn test1(foo: &mut Box<bool>) {
    println!("{:?}", foo)
}

pub fn test2() {
    let foo: &Box<bool>;
}

struct Test3<'a> {
    foo: &'a Box<bool>,
}

trait Test4 {
    fn test4(a: &Box<bool>);
}

impl<'a> Test4 for Test3<'a> {
    fn test4(a: &Box<bool>) {
        unimplemented!();
    }
}

use std::any::Any;

pub fn test5(foo: &mut Box<dyn Any>) {
    println!("{:?}", foo)
}

pub fn test6() {
    let foo: &Box<dyn Any>;
}

struct Test7<'a> {
    foo: &'a Box<dyn Any>,
}

trait Test8 {
    fn test8(a: &Box<dyn Any>);
}

impl<'a> Test8 for Test7<'a> {
    fn test8(a: &Box<dyn Any>) {
        unimplemented!();
    }
}

pub fn test9(foo: &mut Box<dyn Any + Send + Sync>) {
    let _ = foo;
}

pub fn test10() {
    let foo: &Box<dyn Any + Send + 'static>;
}

struct Test11<'a> {
    foo: &'a Box<dyn Any + Send>,
}

trait Test12 {
    fn test4(a: &Box<dyn Any + 'static>);
}

impl<'a> Test12 for Test11<'a> {
    fn test4(a: &Box<dyn Any + 'static>) {
        unimplemented!();
    }
}

fn main() {
    test1(&mut Box::new(false));
    test2();
    test5(&mut (Box::new(false) as Box<dyn Any>));
    test6();
    test9(&mut (Box::new(false) as Box<dyn Any + Send + Sync>));
    test10();
}
