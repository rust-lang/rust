#![warn(clippy::result_large_err)]

fn f() -> Result<(), [u8; 511]> {
    todo!()
}
fn f2() -> Result<(), [u8; 512]> {
    //~^ ERROR: the `Err`-variant returned from this function is very large
    todo!()
}
fn main() {}
