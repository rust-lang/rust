//! Test for <https://github.com/rust-lang/rust/issues/28498>.
//! Example taken from RFC 1238 text
//! <https://github.com/rust-lang/rfcs/blob/master/text/1238-nonparametric-dropck.md>.
//@ run-pass

use std::cell::Cell;

struct Concrete<'a>(#[allow(dead_code)] u32, Cell<Option<&'a Concrete<'a>>>);

fn main() {
    let mut data = Vec::new();
    data.push(Concrete(0, Cell::new(None)));
    data.push(Concrete(0, Cell::new(None)));

    data[0].1.set(Some(&data[1]));
    data[1].1.set(Some(&data[0]));
}
