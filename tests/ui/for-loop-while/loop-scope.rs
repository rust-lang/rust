//@ run-pass

pub fn main() {
    let x = vec![10, 20, 30];
    let mut sum = 0;
    for x in &x { sum += *x; }
    assert_eq!(sum, 60);
}
