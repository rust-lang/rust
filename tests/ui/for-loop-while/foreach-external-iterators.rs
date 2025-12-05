//@ run-pass

pub fn main() {
    let x = [1; 100];
    let mut y = 0;
    for i in &x[..] {
        y += *i
    }
    assert_eq!(y, 100);
}
