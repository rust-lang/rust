//@ run-pass

pub fn main() {
    let mut sum = 0;
    let xs = vec![1, 2, 3, 4, 5];
    for x in &xs {
        sum += *x;
    }
    assert_eq!(sum, 15);
}
