// run-rustfix

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

#[rustfmt::skip]
fn strange_some_no_else(x: &str) -> i32 {
    {
        if let Some(y) = x   .   parse()   .   ok   ()    {
            return y;
        };
        0
    }
}

fn main() {
    let _ = str_to_int("1");
    let _ = str_to_int_ok("2");
    let _ = strange_some_no_else("3");
}
