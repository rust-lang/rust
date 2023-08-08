// run-pass
// We used to have a __rust_abi shim that resulted in duplicated symbols
// whenever the item path wasn't enough to disambiguate between them.
fn main() {
    let a = {
        extern "C" fn good() -> i32 { return 0; }
        good as extern "C" fn() -> i32
    };
    let b = {
        extern "C" fn good() -> i32 { return 5; }
        good as extern "C" fn() -> i32
    };

    assert!(a != b);
    assert_eq!((a(), b()), (0, 5));
}
