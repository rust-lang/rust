#![allow(unused, dead_code)]

fn test_unwrap() -> Option<i32> {
    let b: Result<i32, ()> = Ok(1);
    let v: i32 = b; // return type is not `Result`, we don't suggest ? here
    //~^ ERROR mismatched types
    Some(v)
}

fn test_unwrap_option() -> Result<i32, ()> {
    let b = Some(1);
    let v: i32 = b; // return type is not `Option`, we don't suggest ? here
    //~^ ERROR mismatched types
    Ok(v)
}

fn main() {
    let v: i32 = Some(0); //~ ERROR mismatched types

    let c = Ok(false);
    let v: i32 = c; //~ ERROR mismatched types

}
