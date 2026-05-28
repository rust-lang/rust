//@ run-pass
#![allow(dead_code)]
#![deny(unused_mut)]

#[derive(Debug)]
struct A {}

fn init_a() -> A {
    A {}
}

#[derive(Debug)]
struct B<'a> {
    ed: &'a mut A,
}

fn init_b<'a>(ed: &'a mut A) -> B<'a> {
    B { ed }
}

#[derive(Debug)]
struct C<'a> {
    pd: &'a mut B<'a>,
}

fn init_c<'a>(pd: &'a mut B<'a>) -> C<'a> {
    C { pd }
}

#[derive(Debug)]
struct D<'a> {
    sd: &'a mut C<'a>,
}

fn init_d<'a>(sd: &'a mut C<'a>) -> D<'a> {
    D { sd }
}

fn main() {
    let mut a = init_a();
    let mut b = init_b(&mut a);
    let mut c = init_c(&mut b);

    let d = init_d(&mut c);

    println!("{:?}", d)
}
