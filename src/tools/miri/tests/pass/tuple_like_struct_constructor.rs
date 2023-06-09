fn main() {
    #[derive(PartialEq, Eq, Debug)]
    struct A(i32);
    assert_eq!(Some(42).map(A), Some(A(42)));
}
