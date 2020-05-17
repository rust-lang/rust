#![warn(clippy::unwrap_used)]

fn unwrap_option() {
    let opt = Some(0);
    let _ = opt.unwrap();
}

fn unwrap_result() {
    let res: Result<u8, ()> = Ok(0);
    let _ = res.unwrap();
}

fn main() {
    unwrap_option();
    unwrap_result();
}
