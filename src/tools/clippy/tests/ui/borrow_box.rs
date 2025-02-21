#![deny(clippy::borrowed_box)]
#![allow(dead_code, unused_variables)]
#![allow(
    clippy::uninlined_format_args,
    clippy::disallowed_names,
    clippy::needless_pass_by_ref_mut,
    clippy::needless_lifetimes
)]

use std::fmt::Display;

pub fn test1(foo: &mut Box<bool>) {
    // Although this function could be changed to "&mut bool",
    // avoiding the Box, mutable references to boxes are not
    // flagged by this lint.
    //
    // This omission is intentional: By passing a mutable Box,
    // the memory location of the pointed-to object could be
    // modified. By passing a mutable reference, the contents
    // could change, but not the location.
    println!("{:?}", foo)
}

pub fn test2() {
    let foo: &Box<bool>;
    //~^ borrowed_box
}

struct Test3<'a> {
    foo: &'a Box<bool>,
    //~^ borrowed_box
}

trait Test4 {
    fn test4(a: &Box<bool>);
    //~^ borrowed_box
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

pub fn test13(boxed_slice: &mut Box<[i32]>) {
    // Unconditionally replaces the box pointer.
    //
    // This cannot be accomplished if "&mut [i32]" is passed,
    // and provides a test case where passing a reference to
    // a Box is valid.
    let mut data = vec![12];
    *boxed_slice = data.into_boxed_slice();
}

// The suggestion should include proper parentheses to avoid a syntax error.
pub fn test14(_display: &Box<dyn Display>) {}
//~^ borrowed_box

pub fn test15(_display: &Box<dyn Display + Send>) {}
//~^ borrowed_box

pub fn test16<'a>(_display: &'a Box<dyn Display + 'a>) {}
//~^ borrowed_box

pub fn test17(_display: &Box<impl Display>) {}
//~^ borrowed_box

pub fn test18(_display: &Box<impl Display + Send>) {}
//~^ borrowed_box

pub fn test19<'a>(_display: &'a Box<impl Display + 'a>) {}
//~^ borrowed_box

// This exists only to check what happens when parentheses are already present.
// Even though the current implementation doesn't put extra parentheses,
// it's fine that unnecessary parentheses appear in the future for some reason.
pub fn test20(_display: &Box<(dyn Display + Send)>) {}
//~^ borrowed_box

#[allow(clippy::borrowed_box)]
trait Trait {
    fn f(b: &Box<bool>);
}

// Trait impls are not linted
impl Trait for () {
    fn f(_: &Box<bool>) {}
}

fn main() {
    test1(&mut Box::new(false));
    test2();
    test5(&mut (Box::new(false) as Box<dyn Any>));
    test6();
    test9(&mut (Box::new(false) as Box<dyn Any + Send + Sync>));
    test10();
}
