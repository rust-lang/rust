#![warn(clippy::all)]
#![warn(clippy::redundant_pattern_matching)]
#![allow(clippy::unit_arg, clippy::let_unit_value)]

fn main() {
    if let Ok(_) = Ok::<i32, i32>(42) {}

    if let Err(_) = Err::<i32, i32>(42) {}

    if let None = None::<()> {}

    if let Some(_) = Some(42) {}

    if Ok::<i32, i32>(42).is_ok() {}

    if Err::<i32, i32>(42).is_err() {}

    if None::<i32>.is_none() {}

    if Some(42).is_some() {}

    if let Ok(x) = Ok::<i32, i32>(42) {
        println!("{}", x);
    }

    match Ok::<i32, i32>(42) {
        Ok(_) => true,
        Err(_) => false,
    };

    match Ok::<i32, i32>(42) {
        Ok(_) => false,
        Err(_) => true,
    };

    match Err::<i32, i32>(42) {
        Ok(_) => false,
        Err(_) => true,
    };

    match Err::<i32, i32>(42) {
        Ok(_) => true,
        Err(_) => false,
    };

    match Some(42) {
        Some(_) => true,
        None => false,
    };

    match None::<()> {
        Some(_) => false,
        None => true,
    };

    let _ = match None::<()> {
        Some(_) => false,
        None => true,
    };

    let _ = if let Ok(_) = Ok::<usize, ()>(4) { true } else { false };

    let _ = does_something();
    let _ = returns_unit();

    let opt = Some(false);
    let x = if let Some(_) = opt { true } else { false };
    takes_bool(x);
    let y = if let Some(_) = opt {};
    takes_unit(y);
}

fn takes_bool(x: bool) {}
fn takes_unit(x: ()) {}

fn does_something() -> bool {
    if let Ok(_) = Ok::<i32, i32>(4) {
        true
    } else {
        false
    }
}

fn returns_unit() {
    if let Ok(_) = Ok::<i32, i32>(4) {
        true
    } else {
        false
    };
}
