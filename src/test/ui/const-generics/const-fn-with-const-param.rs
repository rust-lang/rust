// Checks that `const fn` with const params can be used.
// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

const fn const_u32_identity<const X: u32>() -> u32 {
    X
}

fn main() {
    assert_eq!(const_u32_identity::<18>(), 18);
}
