//@run-rustfix
#![warn(clippy::all)]
#![warn(clippy::redundant_pattern_matching)]
#![allow(deprecated, unused_must_use)]
#![allow(
    clippy::if_same_then_else,
    clippy::match_like_matches_macro,
    clippy::needless_bool,
    clippy::uninlined_format_args,
    clippy::unnecessary_wraps
)]

fn main() {
    let result: Result<usize, usize> = Err(5);
    if let Ok(_) = &result {}

    if let Ok(_) = Ok::<i32, i32>(42) {}

    if let Err(_) = Err::<i32, i32>(42) {}

    while let Ok(_) = Ok::<i32, i32>(10) {}

    while let Err(_) = Ok::<i32, i32>(10) {}

    if Ok::<i32, i32>(42).is_ok() {}

    if Err::<i32, i32>(42).is_err() {}

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

    let _ = if let Ok(_) = Ok::<usize, ()>(4) { true } else { false };

    issue5504();
    issue6067();
    issue6065();
    issue10726();
    issue10803();

    let _ = if let Ok(_) = gen_res() {
        1
    } else if let Err(_) = gen_res() {
        2
    } else {
        3
    };
}

fn gen_res() -> Result<(), ()> {
    Ok(())
}

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

fn issue6065() {
    macro_rules! if_let_in_macro {
        ($pat:pat, $x:expr) => {
            if let Some($pat) = $x {}
        };
    }

    // shouldn't be linted
    if_let_in_macro!(_, Some(42));
}

// Methods that are unstable const should not be suggested within a const context, see issue #5697.
// However, in Rust 1.48.0 the methods `is_ok` and `is_err` of `Result` were stabilized as const,
// so the following should be linted.
const fn issue6067() {
    if let Ok(_) = Ok::<i32, i32>(42) {}

    if let Err(_) = Err::<i32, i32>(42) {}

    while let Ok(_) = Ok::<i32, i32>(10) {}

    while let Err(_) = Ok::<i32, i32>(10) {}

    match Ok::<i32, i32>(42) {
        Ok(_) => true,
        Err(_) => false,
    };

    match Err::<i32, i32>(42) {
        Ok(_) => false,
        Err(_) => true,
    };
}

fn issue10726() {
    // This is optional, but it makes the examples easier
    let x: Result<i32, i32> = Ok(42);

    match x {
        Ok(_) => true,
        _ => false,
    };

    match x {
        Ok(_) => false,
        _ => true,
    };

    match x {
        Err(_) => true,
        _ => false,
    };

    match x {
        Err(_) => false,
        _ => true,
    };

    // Don't lint
    match x {
        Err(16) => false,
        _ => true,
    };

    // Don't lint
    match x {
        Ok(16) => false,
        _ => true,
    };
}

fn issue10803() {
    let x: Result<i32, i32> = Ok(42);

    let _ = matches!(x, Ok(_));

    let _ = matches!(x, Err(_));

    // Don't lint
    let _ = matches!(x, Ok(16));

    // Don't lint
    let _ = matches!(x, Err(16));
}
