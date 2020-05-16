// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

fn const_u32_identity<const X: u32>() -> u32 {
    X
}

 fn main() {
    let val = const_u32_identity::<18>();
    assert_eq!(val, 18);
}
