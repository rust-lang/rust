//@ run-pass
#![allow(dead_code)]
// Check that we do not report ambiguities when equivalent predicates
// (modulo bound lifetime names) appears in the environment
// twice. Issue #21965.


fn foo<T>(t: T) -> i32
    where T : for<'a> Fn(&'a u8) -> i32,
          T : for<'b> Fn(&'b u8) -> i32,
{
    t(&3)
}

fn main() {
}
