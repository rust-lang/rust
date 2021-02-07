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
    want::<foo>(f); //~ ERROR arguments to this function are incorrect
    want::<bar>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<usize>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<usize, B>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<foo>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<foo, B>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<bar>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<bar, B>>(f); //~ ERROR arguments to this function are incorrect
}

fn have_foo(f: foo) {
    want::<usize>(f); //~ ERROR arguments to this function are incorrect
    want::<bar>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<usize>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<usize, B>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<foo>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<foo, B>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<bar>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<bar, B>>(f); //~ ERROR arguments to this function are incorrect
}

fn have_foo_foo(f: Foo<foo>) {
    want::<usize>(f); //~ ERROR arguments to this function are incorrect
    want::<foo>(f); //~ ERROR arguments to this function are incorrect
    want::<bar>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<usize>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<usize, B>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<foo, B>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<bar>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<bar, B>>(f); //~ ERROR arguments to this function are incorrect
    want::<&Foo<foo>>(f); //~ ERROR arguments to this function are incorrect
    want::<&Foo<foo, B>>(f); //~ ERROR arguments to this function are incorrect
}

fn have_foo_foo_b(f: Foo<foo, B>) {
    want::<usize>(f); //~ ERROR arguments to this function are incorrect
    want::<foo>(f); //~ ERROR arguments to this function are incorrect
    want::<bar>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<usize>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<usize, B>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<foo>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<bar>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<bar, B>>(f); //~ ERROR arguments to this function are incorrect
    want::<&Foo<foo>>(f); //~ ERROR arguments to this function are incorrect
    want::<&Foo<foo, B>>(f); //~ ERROR arguments to this function are incorrect
}

fn have_foo_foo_b_a(f: Foo<foo, B, A>) {
    want::<usize>(f); //~ ERROR arguments to this function are incorrect
    want::<foo>(f); //~ ERROR arguments to this function are incorrect
    want::<bar>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<usize>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<usize, B>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<foo>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<foo, B>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<bar>>(f); //~ ERROR arguments to this function are incorrect
    want::<Foo<bar, B>>(f); //~ ERROR arguments to this function are incorrect
    want::<&Foo<foo>>(f); //~ ERROR arguments to this function are incorrect
    want::<&Foo<foo, B>>(f); //~ ERROR arguments to this function are incorrect
}

fn main() {}
