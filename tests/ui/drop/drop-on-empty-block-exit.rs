// run-pass
// pretty-expanded FIXME #23616
#![allow(non_camel_case_types)]

enum t { foo(Box<isize>), }

pub fn main() {
    let tt = t::foo(Box::new(10));
    match tt { t::foo(_z) => { } }
}
