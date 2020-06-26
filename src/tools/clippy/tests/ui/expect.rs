#![warn(clippy::expect_used)]

fn expect_option() {
    let opt = Some(0);
    let _ = opt.expect("");
}

fn expect_result() {
    let res: Result<u8, ()> = Ok(0);
    let _ = res.expect("");
}

fn main() {
    expect_option();
    expect_result();
}
