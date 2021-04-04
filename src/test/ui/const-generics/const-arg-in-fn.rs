// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

fn const_u32_identity<const X: u32>() -> u32 {
    X
}

 fn main() {
    let val = const_u32_identity::<18>();
    assert_eq!(val, 18);
}
