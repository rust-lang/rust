#![warn(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    Some(3).unwrap();
    Some(3).expect("Hello world!");

    let a: Result<i32, i32> = Ok(3);
    a.unwrap();
    a.expect("Hello world!");
}
