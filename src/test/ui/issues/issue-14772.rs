// compile-flags: --test

#[test]
mod foo {} //~ ERROR only functions may be used as tests

fn main() {}
