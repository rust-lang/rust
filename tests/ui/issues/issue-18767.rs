// run-pass
// Test that regionck uses the right memcat for patterns in for loops
// and doesn't ICE.


fn main() {
    for &&x in Some(&0_usize).iter() {
        assert_eq!(x, 0)
    }
}
