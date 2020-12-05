// run-pass
#![allow(unused_variables)]
#![allow(unused_assignments)]
#[derive(Debug)]
enum Foo {
    Bar(u32, u32),
    Baz(&'static u32, &'static u32)
}

static NUM: u32 = 100;

fn main () {
    let mut b = Foo::Baz(&NUM, &NUM);
    b = Foo::Bar(f(&b), g(&b));
}

static FNUM: u32 = 1;

fn f (b: &Foo) -> u32 {
    FNUM
}

static GNUM: u32 = 2;

fn g (b: &Foo) -> u32 {
    GNUM
}
