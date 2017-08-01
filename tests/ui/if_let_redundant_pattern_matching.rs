#![feature(plugin)]

#![plugin(clippy)]
#![warn(clippy)]
#![warn(if_let_redundant_pattern_matching)]


fn main() {
    if let Ok(_) = Ok::<i32, i32>(42) {}

    if let Err(_) = Err::<i32, i32>(42) {
    }

    if let None = None::<()> {
    }

    if let Some(_) = Some(42) {
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
