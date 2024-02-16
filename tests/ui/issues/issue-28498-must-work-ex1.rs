//@ run-pass
// Example taken from RFC 1238 text

// https://github.com/rust-lang/rfcs/blob/master/text/1238-nonparametric-dropck.md
//     #examples-of-code-that-must-continue-to-work

use std::cell::Cell;

struct Concrete<'a>(#[allow(dead_code)] u32, Cell<Option<&'a Concrete<'a>>>);

fn main() {
    let mut data = Vec::new();
    data.push(Concrete(0, Cell::new(None)));
    data.push(Concrete(0, Cell::new(None)));

    data[0].1.set(Some(&data[1]));
    data[1].1.set(Some(&data[0]));
}
