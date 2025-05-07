//@ known-bug: #133965
//@ needs-rustc-debug-assertions

struct NonGeneric {}

#[derive(Default)]
struct NonGeneric<'a, const N: usize> {}

pub fn main() {}
