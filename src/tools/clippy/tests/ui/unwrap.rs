#![warn(clippy::unwrap_used)]
#![allow(clippy::unnecessary_literal_unwrap)]

fn unwrap_option() {
    let opt = Some(0);
    let _ = opt.unwrap();
    //~^ unwrap_used
}

fn unwrap_result() {
    let res: Result<u8, u8> = Ok(0);
    let _ = res.unwrap();
    //~^ unwrap_used

    let _ = res.unwrap_err();
    //~^ unwrap_used
}

fn main() {
    unwrap_option();
    unwrap_result();
}
