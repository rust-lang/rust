fn main() unsafe {
    fn foo(_a: uint, _b: uint) : uint::le(_a, _b) {}
    let a: uint = 1u;
    let b: uint = 4u;
    check (uint::le(a, b));
    let c <- a;
    log(debug, foo(c, b));
}
