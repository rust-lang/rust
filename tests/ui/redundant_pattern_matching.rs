// run-rustfix

#![warn(clippy::all)]
#![warn(clippy::redundant_pattern_matching)]
#![allow(clippy::unit_arg, unused_must_use, clippy::needless_bool)]

fn main() {
    if let Ok(_) = Ok::<i32, i32>(42) {}

    if let Err(_) = Err::<i32, i32>(42) {}

    if let None = None::<()> {}

    if let Some(_) = Some(42) {}

    if let Some(_) = Some(42) {
        foo();
    } else {
        bar();
    }

    while let Some(_) = Some(42) {}

    while let None = Some(42) {}

    while let None = None::<()> {}

    while let Ok(_) = Ok::<i32, i32>(10) {}

    while let Err(_) = Ok::<i32, i32>(10) {}

    let mut v = vec![1, 2, 3];
    while let Some(_) = v.pop() {
        foo();
    }

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
}

fn takes_bool(_: bool) {}

fn foo() {}

fn bar() {}

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
