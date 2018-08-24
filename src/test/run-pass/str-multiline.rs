pub fn main() {
    let a: String = "this \
is a test".to_string();
    let b: String =
        "this \
              is \
              another \
              test".to_string();
    assert_eq!(a, "this is a test".to_string());
    assert_eq!(b, "this is another test".to_string());
}
