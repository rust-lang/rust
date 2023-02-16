#![warn(clippy::unnecessary_unwrap)]

fn unwrap_option() {
    let val = Some(1).unwrap();
    let val = Some(1).expect("this never happens");
}

fn unwrap_result() {
    let val = Ok(1).unwrap();
    let val = Err(1).unwrap_err();
    let val = Ok(1).expect("this never happens");
}

fn main() {
    unwrap_option();
    unwrap_result();
}
