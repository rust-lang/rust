// In #88097, the compiler attempted to coerce a closure type to itself via
// a function pointer, which caused an unnecessary error. Check that this
// behavior has been fixed.

//@ check-pass

fn peculiar() -> impl Fn(u8) -> u8 {
    return |x| x + 1
}

fn peculiar2() -> impl Fn(u8) -> u8 {
    return |x| x + 1;
}

fn peculiar3() -> impl Fn(u8) -> u8 {
    let f = |x| x + 1;
    return f
}

fn peculiar4() -> impl Fn(u8) -> u8 {
    let f = |x| x + 1;
    f
}

fn peculiar5() -> impl Fn(u8) -> u8 {
    let f = |x| x + 1;
    let g = |x| x + 2;
    return if true { f } else { g }
}

fn main() {}
