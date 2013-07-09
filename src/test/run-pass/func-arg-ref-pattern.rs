// exec-env:RUST_POISON_ON_FREE=1

// Test argument patterns where we create refs to the inside of `~`
// boxes. Make sure that we don't free the box as we match the
// pattern.

fn getaddr(~ref x: ~uint) -> *uint {
    let addr: *uint = &*x;
    addr
}

fn checkval(~ref x: ~uint) -> uint {
    *x
}

fn main() {
    let obj = ~1;
    let objptr: *uint = &*obj;
    let xptr = getaddr(obj);
    assert_eq!(objptr, xptr);

    let obj = ~22;
    assert_eq!(checkval(obj), 22);
}
