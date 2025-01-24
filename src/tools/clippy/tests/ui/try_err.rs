//@aux-build:proc_macros.rs
#![feature(try_blocks)]
#![deny(clippy::try_err)]
#![allow(
    clippy::unnecessary_wraps,
    clippy::needless_question_mark,
    clippy::needless_return_with_question_mark
)]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

use std::io;
use std::task::Poll;

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

#[inline_macros]
fn calling_macro() -> Result<i32, i32> {
    // macro
    inline!(
        match $(Ok::<_, i32>(5)) {
            Ok(_) => 0,
            Err(_) => Err(1)?,
        }
    );
    // `Err` arg is another macro
    inline!(
        match $(Ok::<_, i32>(5)) {
            Ok(_) => 0,
            Err(_) => Err(inline!(1))?,
        }
    );
    Ok(5)
}

fn main() {
    basic_test().unwrap();
    into_test().unwrap();
    negative_test().unwrap();
    closure_matches_test().unwrap();
    closure_into_test().unwrap();
    calling_macro().unwrap();

    // We don't want to lint in external macros
    external! {
        pub fn try_err_fn() -> Result<i32, i32> {
            let err: i32 = 1;
            // To avoid warnings during rustfix
            if true { Err(err)? } else { Ok(2) }
        }
    }
}

#[inline_macros]
pub fn macro_inside(fail: bool) -> Result<i32, String> {
    if fail {
        Err(inline!(inline!(String::from("aasdfasdfasdfa"))))?;
    }
    Ok(0)
}

pub fn poll_write(n: usize) -> Poll<io::Result<usize>> {
    if n == 0 {
        Err(io::ErrorKind::WriteZero)?
    } else if n == 1 {
        Err(io::Error::new(io::ErrorKind::InvalidInput, "error"))?
    };

    Poll::Ready(Ok(n))
}

pub fn poll_next(ready: bool) -> Poll<Option<io::Result<()>>> {
    if !ready {
        Err(io::ErrorKind::NotFound)?
    }

    Poll::Ready(None)
}

// Tests that `return` is not duplicated
pub fn try_return(x: bool) -> Result<i32, i32> {
    if x {
        return Err(42)?;
    }
    Ok(0)
}

// Test that the lint is suppressed in try block.
pub fn try_block() -> Result<(), i32> {
    let _: Result<_, i32> = try {
        Err(1)?;
    };
    Ok(())
}
