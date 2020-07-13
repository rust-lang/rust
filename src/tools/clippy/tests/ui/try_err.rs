// run-rustfix
// aux-build:macro_rules.rs

#![deny(clippy::try_err)]

#[macro_use]
extern crate macro_rules;

// Tests that a simple case works
// Should flag `Err(err)?`
pub fn basic_test() -> Result<i32, i32> {
    let err: i32 = 1;
    // To avoid warnings during rustfix
    if true {
        Err(err)?;
    }
    Ok(0)
}

// Tests that `.into()` is added when appropriate
pub fn into_test() -> Result<i32, i32> {
    let err: u8 = 1;
    // To avoid warnings during rustfix
    if true {
        Err(err)?;
    }
    Ok(0)
}

// Tests that tries in general don't trigger the error
pub fn negative_test() -> Result<i32, i32> {
    Ok(nested_error()? + 1)
}

// Tests that `.into()` isn't added when the error type
// matches the surrounding closure's return type, even
// when it doesn't match the surrounding function's.
pub fn closure_matches_test() -> Result<i32, i32> {
    let res: Result<i32, i8> = Some(1)
        .into_iter()
        .map(|i| {
            let err: i8 = 1;
            // To avoid warnings during rustfix
            if true {
                Err(err)?;
            }
            Ok(i)
        })
        .next()
        .unwrap();

    Ok(res?)
}

// Tests that `.into()` isn't added when the error type
// doesn't match the surrounding closure's return type.
pub fn closure_into_test() -> Result<i32, i32> {
    let res: Result<i32, i16> = Some(1)
        .into_iter()
        .map(|i| {
            let err: i8 = 1;
            // To avoid warnings during rustfix
            if true {
                Err(err)?;
            }
            Ok(i)
        })
        .next()
        .unwrap();

    Ok(res?)
}

fn nested_error() -> Result<i32, i32> {
    Ok(1)
}

fn main() {
    basic_test().unwrap();
    into_test().unwrap();
    negative_test().unwrap();
    closure_matches_test().unwrap();
    closure_into_test().unwrap();

    // We don't want to lint in external macros
    try_err!();
}

macro_rules! bar {
    () => {
        String::from("aasdfasdfasdfa")
    };
}

macro_rules! foo {
    () => {
        bar!()
    };
}

pub fn macro_inside(fail: bool) -> Result<i32, String> {
    if fail {
        Err(foo!())?;
    }
    Ok(0)
}
