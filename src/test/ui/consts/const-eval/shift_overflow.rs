enum Foo {
    // test that we detect overflows for non-u32 discriminants
    X = 1 << ((u32::max_value() as u64) + 1), //~ ERROR E0080
    Y = 42,
}


fn main() {
}
