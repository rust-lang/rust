use std::backtrace::Backtrace;

// Unfortunately, this cannot be a unit test because that causes problems
// with type-alias-impl-trait (the assert counts as a defining use).
#[test]
fn assert_send_sync() {
    fn assert<T: Send + Sync>() {}

    assert::<Backtrace>();
}
