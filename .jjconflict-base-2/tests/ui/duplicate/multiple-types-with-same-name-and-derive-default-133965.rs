//@ needs-rustc-debug-assertions

struct NonGeneric {}

#[derive(Default)]
struct NonGeneric<'a, const N: usize> {}
//~^ ERROR: the name `NonGeneric` is defined multiple times

pub fn main() {}
