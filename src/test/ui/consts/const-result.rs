// run-pass

fn main() {
    const X: Result<i32, bool> = Ok(32);
    const Y: Result<&i32, &bool> = X.as_ref();

    const IS_OK: bool = X.is_ok();
    assert!(IS_OK);

    const IS_ERR: bool = Y.is_err();
    assert!(!IS_ERR)
}
