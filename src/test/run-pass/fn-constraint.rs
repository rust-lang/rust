fn main() unsafe {
    fn foo(_a: uint, _b: uint) : uint::le(_a, _b) {}
    let a: uint = 1u;
    let b: uint = 4u;
    check (uint::le(a, b));
    log(debug, foo(a, b));
}
