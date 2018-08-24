#[repr(packed(1))] //~ error: the `#[repr(packed(n))]` attribute is experimental
struct Foo(u64);

fn main() {}
