//@ run-pass

pub fn main() {
    let x = [1; 100];
    let mut y = 0;
    for (n,i) in x.iter().enumerate() {
        if n < 10 {
            continue;
        }
        y += *i;
    }
    assert_eq!(y, 90);
}
