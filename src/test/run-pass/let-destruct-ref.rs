pub fn main() {
    let x = ~"hello";
    let ref y = x;
    assert_eq!(x.slice(0, x.len()), y.slice(0, y.len()));
}
