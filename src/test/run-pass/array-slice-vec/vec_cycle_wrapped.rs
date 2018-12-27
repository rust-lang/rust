// run-pass

use std::cell::Cell;

#[derive(Debug)]
struct Refs<'a> {
    v: Vec<Cell<Option<&'a C<'a>>>>,
}

#[derive(Debug)]
struct C<'a> {
    refs: Refs<'a>,
}

impl<'a> Refs<'a> {
    fn new() -> Refs<'a> {
        Refs { v: Vec::new() }
    }
}

impl<'a> C<'a> {
    fn new() -> C<'a> {
        C { refs: Refs::new() }
    }
}

fn f() {
    let (mut c1, mut c2, mut c3);
    c1 = C::new();
    c2 = C::new();
    c3 = C::new();

    c1.refs.v.push(Cell::new(None));
    c1.refs.v.push(Cell::new(None));
    c2.refs.v.push(Cell::new(None));
    c2.refs.v.push(Cell::new(None));
    c3.refs.v.push(Cell::new(None));
    c3.refs.v.push(Cell::new(None));

    c1.refs.v[0].set(Some(&c2));
    c1.refs.v[1].set(Some(&c3));
    c2.refs.v[0].set(Some(&c2));
    c2.refs.v[1].set(Some(&c3));
    c3.refs.v[0].set(Some(&c1));
    c3.refs.v[1].set(Some(&c2));
}

fn main() {
    f();
}
