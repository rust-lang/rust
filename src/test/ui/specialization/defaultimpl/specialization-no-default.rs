#![feature(specialization)]

// Check a number of scenarios in which one impl tries to override another,
// without correctly using `default`.

////////////////////////////////////////////////////////////////////////////////
// Test 1: one layer of specialization, multiple methods, missing `default`
////////////////////////////////////////////////////////////////////////////////

trait Foo {
    fn foo(&self);
    fn bar(&self);
}

impl<T> Foo for T {
    fn foo(&self) {}
    fn bar(&self) {}
}

impl Foo for u8 {}
impl Foo for u16 {
    fn foo(&self) {} //~ ERROR E0520
}
impl Foo for u32 {
    fn bar(&self) {} //~ ERROR E0520
}

////////////////////////////////////////////////////////////////////////////////
// Test 2: one layer of specialization, missing `default` on associated type
////////////////////////////////////////////////////////////////////////////////

trait Bar {
    type T;
}

impl<T> Bar for T {
    type T = u8;
}

impl Bar for u8 {
    type T = (); //~ ERROR E0520
}

////////////////////////////////////////////////////////////////////////////////
// Test 3a: multiple layers of specialization, missing interior `default`
////////////////////////////////////////////////////////////////////////////////

trait Baz {
    fn baz(&self);
}

default impl<T> Baz for T {
    fn baz(&self) {}
}

impl<T: Clone> Baz for T {
    fn baz(&self) {}
}

impl Baz for i32 {
    fn baz(&self) {} //~ ERROR E0520
}

////////////////////////////////////////////////////////////////////////////////
// Test 3b: multiple layers of specialization, missing interior `default`,
// redundant `default` in bottom layer.
////////////////////////////////////////////////////////////////////////////////

trait Redundant {
    fn redundant(&self);
}

default impl<T> Redundant for T {
    fn redundant(&self) {}
}

impl<T: Clone> Redundant for T {
    fn redundant(&self) {}
}

default impl Redundant for i32 {
    fn redundant(&self) {} //~ ERROR E0520
}

fn main() {}
