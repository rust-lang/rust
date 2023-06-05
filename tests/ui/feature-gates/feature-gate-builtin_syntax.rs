struct Foo {
    v: u8,
    w: u8,
}
fn main() {
    builtin # offset_of(Foo, v); //~ ERROR `builtin #` syntax is unstable
}
