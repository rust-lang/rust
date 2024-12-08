//@ run-rustfix
#![allow(unused, dead_code)]

fn func() -> Option<i32> {
    Some(1)
}

fn test_unwrap() -> Result<i32, ()> {
    let b: Result<i32, ()> = Ok(1);
    let v: i32 = b; //~ ERROR mismatched types
    Ok(v)
}

fn test_unwrap_option() -> Option<i32> {
    let b = Some(1);
    let v: i32 = b; //~ ERROR mismatched types
    Some(v)
}

fn main() {
    let a = Some(1);
    let v: i32 = a; //~ ERROR mismatched types

    let b: Result<i32, ()> = Ok(1);
    let v: i32 = b; //~ ERROR mismatched types

    let v: i32 = func(); //~ ERROR mismatched types

    let a = None;
    let v: i32 = a; //~ ERROR mismatched types
}
