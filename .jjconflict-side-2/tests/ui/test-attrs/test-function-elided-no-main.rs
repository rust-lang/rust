//! Test that #[test] functions are elided when not running tests, causing missing main error

#[test]
fn main() {
    // This function would normally serve as main, but since it's marked with #[test],
    // it gets elided when not running tests
}
//~^ ERROR `main` function not found in crate `test_function_elided_no_main`
