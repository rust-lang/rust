#![feature(plugin)]

#![plugin(clippy)]
#![deny(clippy)]
#![deny(if_let_redundant_pattern_matching)]


fn main() {
    if let Ok(_) = Ok::<i32, i32>(42) {}
    //~^ERROR redundant pattern matching, consider using `is_ok()`
    //~| HELP try this
    //~| SUGGESTION if Ok::<i32, i32>(42).is_ok() {

    if let Err(_) = Err::<i32, i32>(42) {
    //~^ERROR redundant pattern matching, consider using `is_err()`
    //~| HELP try this
    //~| SUGGESTION if Err::<i32, i32>(42).is_err() {
    }

    if let None = None::<()> {
    //~^ERROR redundant pattern matching, consider using `is_none()`
    //~| HELP try this
    //~| SUGGESTION if None::<()>.is_none() {
    }

    if let Some(_) = Some(42) {
    //~^ERROR redundant pattern matching, consider using `is_some()`
    //~| HELP try this
    //~| SUGGESTION if Some(42).is_some() {
    }

    if Ok::<i32, i32>(42).is_ok() {

    }

    if Err::<i32, i32>(42).is_err() {

    }

    if None::<i32>.is_none() {

    }

    if Some(42).is_some() {

    }

    if let Ok(x) = Ok::<i32,i32>(42) {
        println!("{}", x);
    }
}


