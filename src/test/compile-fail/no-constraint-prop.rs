// error-pattern:unsatisfied precondition constraint (for example, uint::le

fn main() unsafe {
    fn foo(_a: uint, _b: uint) : uint::le(_a, _b) {}
    let a: uint = 1u;
    let b: uint = 4u;
    let c: uint = 5u;
    // make sure that the constraint le(b, a) exists...
    check (uint::le(b, a));
    // ...invalidate it...
    b += 1u;
    check (uint::le(c, a));
    // ...and check that it doesn't get set in the poststate of
    // the next statement, since it's not true in the
    // prestate.
    let d <- a;
    log(debug, foo(b, d));
}
