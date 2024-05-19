#![warn(clippy::expect_used)]
#![allow(clippy::unnecessary_literal_unwrap)]

fn expect_option() {
    let opt = Some(0);
    let _ = opt.expect("");
    //~^ ERROR: used `expect()` on an `Option` value
}

fn expect_result() {
    let res: Result<u8, u8> = Ok(0);
    let _ = res.expect("");
    //~^ ERROR: used `expect()` on a `Result` value
    let _ = res.expect_err("");
    //~^ ERROR: used `expect_err()` on a `Result` value
}

fn main() {
    expect_option();
    expect_result();
}
