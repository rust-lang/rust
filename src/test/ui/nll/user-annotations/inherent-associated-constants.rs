#![feature(nll)]

struct A<'a>(&'a ());

impl A<'static> {
    const IC: i32 = 10;
}

fn non_wf_associated_const<'a>(x: i32) {
    A::<'a>::IC; //~ ERROR lifetime may not live long enough
}

fn wf_associated_const<'a>(x: i32) {
    A::<'static>::IC;
}

fn main() {}
