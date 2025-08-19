//! Regression test for https://github.com/rust-lang/rust/issues/11205

//@ run-pass

#![allow(dead_code)]

trait Foo { fn dummy(&self) { } }
impl Foo for isize {}
fn foo(_: [&dyn Foo; 2]) {}
fn foos(_: &[&dyn Foo]) {}
fn foog<T>(_: &[T], _: &[T]) {}

fn bar(_: [Box<dyn Foo>; 2]) {}
fn bars(_: &[Box<dyn Foo+'static>]) {}

fn main() {
    let x: [&dyn Foo; 2] = [&1, &2];
    foo(x);
    foo([&1, &2]);

    let r = &1;
    let x: [&dyn Foo; 2] = [r; 2];
    foo(x);
    foo([&1; 2]);

    let x: &[&dyn Foo] = &[&1, &2];
    foos(x);
    foos(&[&1, &2]);

    let x: &[&dyn Foo] = &[&1, &2];
    let r = &1;
    foog(x, &[r]);

    let x: [Box<dyn Foo>; 2] = [Box::new(1), Box::new(2)];
    bar(x);
    bar([Box::new(1), Box::new(2)]);

    let x: &[Box<dyn Foo+'static>] = &[Box::new(1), Box::new(2)];
    bars(x);
    bars(&[Box::new(1), Box::new(2)]);

    let x: &[Box<dyn Foo+'static>] = &[Box::new(1), Box::new(2)];
    foog(x, &[Box::new(1)]);

    struct T<'a> {
        t: [&'a (dyn Foo+'a); 2]
    }
    let _n = T {
        t: [&1, &2]
    };
    let r = &1;
    let _n = T {
        t: [r; 2]
    };
    let x: [&dyn Foo; 2] = [&1, &2];
    let _n = T {
        t: x
    };

    struct F<'b> {
        t: &'b [&'b (dyn Foo+'b)]
    }
    let _n = F {
        t: &[&1, &2]
    };
    let r = &1;
    let r: [&dyn Foo; 2] = [r; 2];
    let _n = F {
        t: &r
    };
    let x: [&dyn Foo; 2] = [&1, &2];
    let _n = F {
        t: &x
    };

    struct M<'a> {
        t: &'a [Box<dyn Foo+'static>]
    }
    let _n = M {
        t: &[Box::new(1), Box::new(2)]
    };
    let x: [Box<dyn Foo>; 2] = [Box::new(1), Box::new(2)];
    let _n = M {
        t: &x
    };
}
