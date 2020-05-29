// run-pass
#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

const fn const_u32_identity<const X: u32>() -> u32 {
    X
}

fn main() {
    assert_eq!(const_u32_identity::<18>(), 18);
}
