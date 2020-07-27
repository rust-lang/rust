// run-rustfix

#![warn(clippy::all)]
#![warn(clippy::redundant_pattern_matching)]
#![allow(
    clippy::unit_arg,
    unused_must_use,
    clippy::needless_bool,
    clippy::match_like_matches_macro,
    deprecated
)]

fn main() {
    let result: Result<usize, usize> = Err(5);
    if let Ok(_) = &result {}

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

    let opt = Some(false);
    let x = if let Some(_) = opt { true } else { false };
    takes_bool(x);

    issue5504();
    issue5697();

    let _ = if let Some(_) = gen_opt() {
        1
    } else if let None = gen_opt() {
        2
    } else if let Ok(_) = gen_res() {
        3
    } else if let Err(_) = gen_res() {
        4
    } else {
        5
    };
}

fn gen_opt() -> Option<()> {
    None
}

fn gen_res() -> Result<(), ()> {
    Ok(())
}

fn takes_bool(_: bool) {}

fn foo() {}

fn bar() {}

macro_rules! m {
    () => {
        Some(42u32)
    };
}

fn issue5504() {
    fn result_opt() -> Result<Option<i32>, i32> {
        Err(42)
    }

    fn try_result_opt() -> Result<i32, i32> {
        while let Some(_) = r#try!(result_opt()) {}
        if let Some(_) = r#try!(result_opt()) {}
        Ok(42)
    }

    try_result_opt();

    if let Some(_) = m!() {}
    while let Some(_) = m!() {}
}

// None of these should be linted because none of the suggested methods
// are `const fn` without toggling a feature.
const fn issue5697() {
    if let Ok(_) = Ok::<i32, i32>(42) {}

    if let Err(_) = Err::<i32, i32>(42) {}

    if let Some(_) = Some(42) {}

    if let None = None::<()> {}

    while let Ok(_) = Ok::<i32, i32>(10) {}

    while let Err(_) = Ok::<i32, i32>(10) {}

    while let Some(_) = Some(42) {}

    while let None = None::<()> {}

    match Ok::<i32, i32>(42) {
        Ok(_) => true,
        Err(_) => false,
    };

    match Err::<i32, i32>(42) {
        Ok(_) => false,
        Err(_) => true,
    };
    match Some(42) {
        Some(_) => true,
        None => false,
    };

    match None::<()> {
        Some(_) => false,
        None => true,
    };
}
