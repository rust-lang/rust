//@ run-pass

pub fn main() {
    let x = [1; 100];
    let y = [2; 100];
    let mut p = 0;
    let mut q = 0;
    for i in &x[..] {
        for j in &y[..] {
            p += *j;
        }
        q += *i + p;
    }
    assert_eq!(q, 1010100);
}
