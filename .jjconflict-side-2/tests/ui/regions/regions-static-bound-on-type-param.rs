// https://github.com/rust-lang/rust/issues/162882

use std::error::Error;
fn err_is<T: Error + 'static>(err: &dyn Error) -> bool {
    err.is::<T>() //~ ERROR: borrowed data escapes outside of function
}
fn main() {}
