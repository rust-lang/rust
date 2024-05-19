#![warn(clippy::unwrap_used)]
#![allow(clippy::unnecessary_literal_unwrap)]

fn unwrap_option() {
    let opt = Some(0);
    let _ = opt.unwrap();
    //~^ ERROR: used `unwrap()` on an `Option` value
}

fn unwrap_result() {
    let res: Result<u8, u8> = Ok(0);
    let _ = res.unwrap();
    //~^ ERROR: used `unwrap()` on a `Result` value
    let _ = res.unwrap_err();
    //~^ ERROR: used `unwrap_err()` on a `Result` value
}

fn main() {
    unwrap_option();
    unwrap_result();
}
