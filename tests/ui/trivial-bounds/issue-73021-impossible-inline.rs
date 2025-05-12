//@ build-pass
//@ revisions: no-opt inline
//@ [inline]compile-flags: -Zmir-opt-level=3 --emit=mir
#![feature(trivial_bounds)]
#![allow(unused)]

trait Foo {
    fn test(&self);
}

fn foo<'a>(s: &'a mut ())
where
    &'a mut (): Foo,
{
    s.test();
}

fn clone(it: &mut ()) -> &mut ()
where
    for<'any> &'any mut (): Clone,
    //~^ WARN trait bound for<'any> &'any mut (): Clone does not depend on any type or lifetime parameters
{
    it.clone()
}

fn generic_function<X: Foo>(x: X) {}

struct S where i32: Foo;
//~^ WARN trait bound i32: Foo does not depend on any type or lifetime parameters

impl Foo for () where i32: Foo {
//~^ WARN trait bound i32: Foo does not depend on any type or lifetime parameters
    fn test(&self) {
        3i32.test();
        Foo::test(&4i32);
        generic_function(5i32);
    }
}

fn f() where i32: Foo {
//~^ WARN trait bound i32: Foo does not depend on any type or lifetime parameters
    let s = S;
    3i32.test();
    Foo::test(&4i32);
    generic_function(5i32);
}

fn g() where &'static str: Foo {
//~^ WARN trait bound &'static str: Foo does not depend on any type or lifetime parameters
    "Foo".test();
    Foo::test(&"Foo");
    generic_function("Foo");
}

fn use_op(s: String) -> String
where
    String: ::std::ops::Neg<Output = String>,
//~^ WARN trait bound String: Neg does not depend on any type or lifetime parameters
{
    -s
}

fn use_for()
where
    i32: Iterator,
//~^ WARN trait bound i32: Iterator does not depend on any type or lifetime parameters
{
    for _ in 2i32 {}
}

fn main() {}
