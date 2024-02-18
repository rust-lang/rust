//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// Test what happens when a HR obligation is applied to an impl with
// "outlives" bounds. Currently we're pretty conservative here; this
// will probably improve in time.

trait Foo<X> {
    fn foo(&self, x: X) { }
}

fn want_foo<T>()
    where T : for<'a> Foo<&'a isize>
{
}

// Expressed as a where clause

struct SomeStruct<X> {
    x: X
}

impl<'a,X> Foo<&'a isize> for SomeStruct<X>
    where X : 'a
{
}

fn one() {
    want_foo::<SomeStruct<usize>>();
}

// Expressed as shorthand

struct AnotherStruct<X> {
    x: X
}

impl<'a,X:'a> Foo<&'a isize> for AnotherStruct<X>
{
}

fn two() {
    want_foo::<AnotherStruct<usize>>();
}

fn main() { }
