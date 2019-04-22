#![allow(unused_comparisons)]
// Test that you only need the syntax gate if you don't mention the structs.
// (Obsoleted since both features are stabilized)

fn main() {
    let mut count = 0;
    for i in 0_usize..=10 {
        assert!(i >= 0 && i <= 10);
        count += i;
    }
    assert_eq!(count, 55);
}
