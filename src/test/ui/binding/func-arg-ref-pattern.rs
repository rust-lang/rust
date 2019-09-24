// run-pass
// exec-env:RUST_POISON_ON_FREE=1

// Test argument patterns where we create refs to the inside of
// boxes. Make sure that we don't free the box as we match the
// pattern.

#![feature(box_patterns)]
#![feature(box_syntax)]

fn getaddr(box ref x: Box<usize>) -> *const usize {
    let addr: *const usize = &*x;
    addr
}

fn checkval(box ref x: Box<usize>) -> usize {
    *x
}

pub fn main() {
    let obj: Box<_> = box 1;
    let objptr: *const usize = &*obj;
    let xptr = getaddr(obj);
    assert_eq!(objptr, xptr);

    let obj = box 22;
    assert_eq!(checkval(obj), 22);
}
