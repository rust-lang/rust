#![allow(non_camel_case_types)]

trait Qux {}
struct A;
struct B;
impl Qux for A {}
impl Qux for B {}

struct Foo<T, U: Qux = A, V: Qux = B>(T, U, V);

struct foo;
struct bar;

fn want<T>(t: T) {}

fn have_usize(f: usize) {
    want::<foo>(f); //~ ERROR mismatched types
    want::<bar>(f); //~ ERROR mismatched types
    want::<Foo<usize>>(f); //~ ERROR mismatched types
    want::<Foo<usize, B>>(f); //~ ERROR mismatched types
    want::<Foo<foo>>(f); //~ ERROR mismatched types
    want::<Foo<foo, B>>(f); //~ ERROR mismatched types
    want::<Foo<bar>>(f); //~ ERROR mismatched types
    want::<Foo<bar, B>>(f); //~ ERROR mismatched types
}

fn have_foo(f: foo) {
    want::<usize>(f); //~ ERROR mismatched types
    want::<bar>(f); //~ ERROR mismatched types
    want::<Foo<usize>>(f); //~ ERROR mismatched types
    want::<Foo<usize, B>>(f); //~ ERROR mismatched types
    want::<Foo<foo>>(f); //~ ERROR mismatched types
    want::<Foo<foo, B>>(f); //~ ERROR mismatched types
    want::<Foo<bar>>(f); //~ ERROR mismatched types
    want::<Foo<bar, B>>(f); //~ ERROR mismatched types
}

fn have_foo_foo(f: Foo<foo>) {
    want::<usize>(f); //~ ERROR mismatched types
    want::<foo>(f); //~ ERROR mismatched types
    want::<bar>(f); //~ ERROR mismatched types
    want::<Foo<usize>>(f); //~ ERROR mismatched types
    want::<Foo<usize, B>>(f); //~ ERROR mismatched types
    want::<Foo<foo, B>>(f); //~ ERROR mismatched types
    want::<Foo<bar>>(f); //~ ERROR mismatched types
    want::<Foo<bar, B>>(f); //~ ERROR mismatched types
    want::<&Foo<foo>>(f); //~ ERROR mismatched types
    want::<&Foo<foo, B>>(f); //~ ERROR mismatched types
}

fn have_foo_foo_b(f: Foo<foo, B>) {
    want::<usize>(f); //~ ERROR mismatched types
    want::<foo>(f); //~ ERROR mismatched types
    want::<bar>(f); //~ ERROR mismatched types
    want::<Foo<usize>>(f); //~ ERROR mismatched types
    want::<Foo<usize, B>>(f); //~ ERROR mismatched types
    want::<Foo<foo>>(f); //~ ERROR mismatched types
    want::<Foo<bar>>(f); //~ ERROR mismatched types
    want::<Foo<bar, B>>(f); //~ ERROR mismatched types
    want::<&Foo<foo>>(f); //~ ERROR mismatched types
    want::<&Foo<foo, B>>(f); //~ ERROR mismatched types
}

fn have_foo_foo_b_a(f: Foo<foo, B, A>) {
    want::<usize>(f); //~ ERROR mismatched types
    want::<foo>(f); //~ ERROR mismatched types
    want::<bar>(f); //~ ERROR mismatched types
    want::<Foo<usize>>(f); //~ ERROR mismatched types
    want::<Foo<usize, B>>(f); //~ ERROR mismatched types
    want::<Foo<foo>>(f); //~ ERROR mismatched types
    want::<Foo<foo, B>>(f); //~ ERROR mismatched types
    want::<Foo<bar>>(f); //~ ERROR mismatched types
    want::<Foo<bar, B>>(f); //~ ERROR mismatched types
    want::<&Foo<foo>>(f); //~ ERROR mismatched types
    want::<&Foo<foo, B>>(f); //~ ERROR mismatched types
}

fn main() {}
