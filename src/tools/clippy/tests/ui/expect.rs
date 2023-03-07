#![warn(clippy::expect_used)]

fn expect_option() {
    let opt = Some(0);
    let _ = opt.expect("");
}

fn expect_result() {
    let res: Result<u8, u8> = Ok(0);
    let _ = res.expect("");
    let _ = res.expect_err("");
}

fn main() {
    expect_option();
    expect_result();
}
