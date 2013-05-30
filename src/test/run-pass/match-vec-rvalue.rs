// Tests that matching rvalues with drops does not crash.

fn main() {
    match ~[1, 2, 3] {
        x => {
            assert_eq!(x.len(), 3);
            assert_eq!(x[0], 1);
            assert_eq!(x[1], 2);
            assert_eq!(x[2], 3);
        }
    }
}
