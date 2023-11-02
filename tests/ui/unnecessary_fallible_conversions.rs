#![warn(clippy::unnecessary_fallible_conversions)]

fn main() {
    let _: i64 = 0i32.try_into().unwrap();
    let _: i64 = 0i32.try_into().expect("can't happen");
}
