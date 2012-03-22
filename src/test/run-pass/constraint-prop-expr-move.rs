
fn main() unsafe {
    fn foo(_a: uint, _b: uint) : uint::le(_a, _b) {}
    let a: uint = 1u;
    let b: uint = 4u;
    let mut c: uint = 17u;
    check (uint::le(a, b));
    c <- a;
    log(debug, foo(c, b));
}
