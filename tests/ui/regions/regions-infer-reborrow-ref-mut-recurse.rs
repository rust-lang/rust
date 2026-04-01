//@ run-pass
#![allow(dead_code)]
// Test an edge case in region inference: the lifetime of the borrow
// of `*x` must be extended to at least 'a.


fn foo<'a,'b>(x: &'a &'b mut isize) -> &'a isize {
    let y = &*x; // should be inferred to have type &'a &'b mut isize...

    // ...because if we inferred, say, &'x &'b mut isize where 'x <= 'a,
    // this reborrow would be illegal:
    &**y
}

pub fn main() {
    /* Just want to know that it compiles. */
}
