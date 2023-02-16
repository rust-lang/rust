#![warn(clippy::unnecessary_literal_unwrap)]

fn unwrap_option() {
    let val = Some(1).unwrap();
    let val = Some(1).expect("this never happens");
}

fn main() {
    unwrap_option();
}
