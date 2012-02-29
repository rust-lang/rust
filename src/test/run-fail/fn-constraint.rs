// error-pattern:Predicate uint::le(a, b) failed

fn main() unsafe {
    fn foo(_a: uint, _b: uint) : uint::le(_a, _b) {}
    let a: uint = 4u;
    let b: uint = 1u;
    check (uint::le(a, b));
    log(error, foo(a, b));
}
