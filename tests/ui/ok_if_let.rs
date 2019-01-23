#![warn(clippy::if_let_some_result)]

fn str_to_int(x: &str) -> i32 {
    if let Some(y) = x.parse().ok() {
        y
    } else {
        0
    }
}

fn str_to_int_ok(x: &str) -> i32 {
    if let Ok(y) = x.parse() {
        y
    } else {
        0
    }
}

fn main() {
    let y = str_to_int("1");
    let z = str_to_int_ok("2");
    println!("{}{}", y, z);
}
