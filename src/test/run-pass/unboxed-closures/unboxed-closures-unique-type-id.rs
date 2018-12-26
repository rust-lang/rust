// run-pass

// This code used to produce the following ICE:
//
//    error: internal compiler error: get_unique_type_id_of_type() -
//    unexpected type: closure,
//    Closure(syntax::ast::DefId{krate: 0, node: 66},
//    ReScope(63))
//
// This is a regression test for issue #17021.
//
// compile-flags: -g

use std::ptr;

pub fn replace_map<'a, T, F>(src: &mut T, prod: F) where F: FnOnce(T) -> T {
    unsafe { *src = prod(ptr::read(src as *mut T as *const T)); }
}

pub fn main() {
    let mut a = 7;
    let b = &mut a;
    replace_map(b, |x: usize| x * 2);
    assert_eq!(*b, 14);
}
