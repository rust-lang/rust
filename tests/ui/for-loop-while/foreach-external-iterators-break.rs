// run-pass

pub fn main() {
    let x = [1; 100];
    let mut y = 0;
    for i in &x[..] {
        if y > 10 {
            break;
        }
        y += *i;
    }
    assert_eq!(y, 11);
}
