#![allow(non_camel_case_types)]
#![allow(dead_code)]
// pretty-expanded FIXME #23616

use std::ptr;

// check that we do not report a type like this as uninstantiable,
// even though it would be if the nxt field had type @foo:
struct foo(X);

struct X { x: usize, nxt: *const foo }

pub fn main() {
    let _x = foo(X {x: 0, nxt: ptr::null()});
}
