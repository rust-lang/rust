//@ run-pass
fn const_u32_identity<const X: u32>() -> u32 {
    X
}

 fn main() {
    let val = const_u32_identity::<18>();
    assert_eq!(val, 18);
}
