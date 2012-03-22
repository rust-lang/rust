fn main() unsafe {
    fn foo(_a: uint, _b: uint) : uint::le(_a, _b) {}
    let mut a: uint = 4u;
    let mut b: uint = 1u;
    check (uint::le(b, a));
    b <-> a;
    log(debug, foo(a, b));
}
