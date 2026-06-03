// Tests below represent situations when type of the first argument can not be adjusted
// to the type of first parameter (i.e., Rc<T> -> &mut T).

#![feature(fn_delegation)]

use std::sync::Arc;
use std::pin::Pin;
use std::rc::Rc;

trait Trait: Sized {
    fn by_value(self) -> i32 { 1 }
    fn by_mut_ref(&mut self) -> i32 { 2 }
    fn by_ref(&self) -> i32 { 3 }
    fn r#box(self: Box<Self>) -> i32 { 4 }
    fn arc(self: Arc<Self>) -> i32 { 5 }
    fn rc(self: Rc<Self>) -> i32 { 6 }
    fn pin_box(self: Pin<Box<Self>>) -> i32 { 7 }
    fn pin_rc(self: Pin<Rc<Self>>) -> i32 { 8 }
    fn pin_arc(self: Pin<Arc<Self>>) -> i32 { 9 }
    fn box_box(self: Box<Box<Self>>) -> i32 { 10 }
}

struct F;
impl Trait for F {}

struct S(F);

fn foo() -> F {
    F
}

impl S {
    reuse Trait::{by_value, by_mut_ref, by_ref} {
        println!("123");
        let x = &self.0;
        foo()
    }
}

struct S1(F);

impl S1 {
    reuse Trait::{by_value, by_mut_ref, by_ref} {
        println!("123");
        let x = &self.0;
        foo
        //~^ ERROR: mismatched types
        //~| ERROR: mismatched types
        //~| ERROR: the trait bound `fn() -> F {foo}: Trait` is not satisfied
    }
}

struct S2(F);

impl S2 {
    reuse Trait::{by_value, by_mut_ref, by_ref} {
        println!("123");
        let x = &self.0;
        let x = foo();

        x
        //~^ ERROR: cannot borrow `x` as mutable, as it is not declared as mutable
    }
}

struct S3(F);

impl S3 {
    reuse Trait::{by_value, by_mut_ref, by_ref} {
        println!("123");
        let x = &self.0;
        let x = foo();

        &mut x
        //~^ ERROR: cannot borrow `x` as mutable, as it is not declared as mutable
        //~| ERROR: cannot borrow `x` as mutable, as it is not declared as mutable
        //~| ERROR: cannot borrow `x` as mutable, as it is not declared as mutable
        //~| ERROR: cannot move out of a mutable reference
    }
}

struct X(F);

impl X {
    reuse Trait::* { &mut self.0 }
    //~^ ERROR: cannot borrow `self.0` as mutable, as it is behind a `&` reference
    //~| ERROR: cannot borrow `self.0` as mutable, as `self` is not declared as mutable
    //~| ERROR: cannot move out of a mutable reference
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
}

struct X1(F);

impl X1 {
    reuse Trait::* { &self.0 }
    //~^ ERROR: cannot borrow data in a `&` reference as mutable
    //~| ERROR: cannot move out of a shared reference
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
}

struct X2(F);

impl X2 {
    reuse Trait::* { &&&&self.0 }
    //~^ ERROR: cannot move out of a shared reference
    //~| ERROR: cannot borrow data in a `&` reference as mutable
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
}

struct X3(Box<F>);

impl X3 {
    reuse Trait::* { self.0.as_ref() }
    //~^ ERROR: cannot borrow data in a `&` reference as mutable
    //~| ERROR: cannot move out of a shared reference
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
}

struct X4(F);

impl X4 {
    reuse Trait::* { &mut &mut &mut self.0 }
    //~^ ERROR: cannot move out of a mutable reference
    //~| ERROR: cannot borrow `self.0` as mutable, as `self` is not declared as mutable
    //~| ERROR: cannot borrow `self.0` as mutable, as it is behind a `&` reference
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
}

struct X5(F);

impl X5 {
    reuse Trait::* { &&mut self.0 }
    //~^ ERROR: cannot borrow `self.0` as mutable, as it is behind a `&` reference
    //~| ERROR: cannot borrow data in a `&` reference as mutable [E0596]
    //~| ERROR: cannot borrow `self.0` as mutable, as `self` is not declared as mutable
    //~| ERROR: cannot move out of a shared reference
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
}

struct X6(Box<F>);

impl X6 {
    reuse Trait::* { self.0 }
    //~^ ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
}

struct X7(Box<Arc<Box<F>>>);

impl X7 {
    reuse Trait::* { self.0 }
    //~^ ERROR: cannot borrow data in an `Arc` as mutable
    //~| ERROR: cannot move out of an `Arc`
    //~| ERROR: cannot move out of an `Arc`
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
}

struct X8(Pin<Box<F>>);

impl X8 {
    reuse Trait::* { self.0 }
    //~^ ERROR: cannot move out of dereference of `Pin<Box<F>>`
    //~| ERROR: cannot borrow data in dereference of `Pin<Box<F>>` as mutable
    //~| ERROR: cannot move out of dereference of `Pin<Box<X8>>`
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
}

struct OtherStruct;
struct X9(OtherStruct);

impl X9 {
    reuse Trait::* { self.0 }
    //~^ ERROR: the trait bound `OtherStruct: Trait` is not satisfied
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
}

fn main() {}
