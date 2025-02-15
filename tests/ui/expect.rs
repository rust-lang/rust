#![warn(clippy::expect_used)]
#![allow(clippy::unnecessary_literal_unwrap)]

fn expect_option() {
    let opt = Some(0);
    let _ = opt.expect("");
    //~^ expect_used
}

fn expect_result() {
    let res: Result<u8, u8> = Ok(0);
    let _ = res.expect("");
    //~^ expect_used

    let _ = res.expect_err("");
    //~^ expect_used
}

fn main() {
    expect_option();
    expect_result();
}
