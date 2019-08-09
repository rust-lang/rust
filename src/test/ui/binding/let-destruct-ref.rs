// run-pass

pub fn main() {
    let x = 3_usize;
    let ref y = x;
    assert_eq!(x, *y);
}
