// run-pass
fn main() {
    let f = |_||x, y| x+y;
    assert_eq!(f(())(1, 2), 3);
}
