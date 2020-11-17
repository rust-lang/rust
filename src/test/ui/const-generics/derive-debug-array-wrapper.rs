// Check that deriving debug on struct with const is permitted.
// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

#[derive(Debug)]
struct X<const N: usize> {
    a: [u32; N],
}

fn main() {}
