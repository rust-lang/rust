fn main() {
    let v1 = mylib_v1::my_macro!();
    assert_eq!(v1, "version 1");

    let v2 = mylib_v2::my_macro!();
    assert_eq!(v2, "version 2");
}
