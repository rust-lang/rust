fn main() {
    let x = 5;
    assert_eq!(Some(&x).map(Some), Some(Some(&x)));
}
