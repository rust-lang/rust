// error-pattern:Predicate uint::le(a, b) failed

fn main() unsafe {
    let a: uint = 4u;
    let b: uint = 1u;
    check (uint::le(a, b));
    log(error, str::unsafe::slice_bytes_safe_range("kitties", a, b));
}
