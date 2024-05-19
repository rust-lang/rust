//@no-rustfix: overlapping suggestions
#![feature(lint_reasons)]
#![deny(clippy::panicking_unwrap, clippy::unnecessary_unwrap)]
#![allow(
    clippy::if_same_then_else,
    clippy::branches_sharing_code,
    clippy::unnecessary_literal_unwrap
)]

macro_rules! m {
    ($a:expr) => {
        if $a.is_some() {
            // unnecessary
            $a.unwrap();
        }
    };
}

macro_rules! checks_in_param {
    ($a:expr, $b:expr) => {
        if $a {
            $b;
        }
    };
}

macro_rules! checks_unwrap {
    ($a:expr, $b:expr) => {
        if $a.is_some() {
            $b;
        }
    };
}

macro_rules! checks_some {
    ($a:expr, $b:expr) => {
        if $a {
            $b.unwrap();
        }
    };
}

fn main() {
    let x = Some(());
    if x.is_some() {
        // unnecessary
        x.unwrap();
        //~^ ERROR: called `unwrap` on `x` after checking its variant with `is_some`
        // unnecessary
        x.expect("an error message");
        //~^ ERROR: called `expect` on `x` after checking its variant with `is_some`
    } else {
        // will panic
        x.unwrap();
        //~^ ERROR: this call to `unwrap()` will always panic
        // will panic
        x.expect("an error message");
        //~^ ERROR: this call to `expect()` will always panic
    }
    if x.is_none() {
        // will panic
        x.unwrap();
        //~^ ERROR: this call to `unwrap()` will always panic
    } else {
        // unnecessary
        x.unwrap();
        //~^ ERROR: called `unwrap` on `x` after checking its variant with `is_none`
    }
    m!(x);
    // ok
    checks_in_param!(x.is_some(), x.unwrap());
    // ok
    checks_unwrap!(x, x.unwrap());
    // ok
    checks_some!(x.is_some(), x);
    let mut x: Result<(), ()> = Ok(());
    if x.is_ok() {
        // unnecessary
        x.unwrap();
        //~^ ERROR: called `unwrap` on `x` after checking its variant with `is_ok`
        // unnecessary
        x.expect("an error message");
        //~^ ERROR: called `expect` on `x` after checking its variant with `is_ok`
        // will panic
        x.unwrap_err();
        //~^ ERROR: this call to `unwrap_err()` will always panic
    } else {
        // will panic
        x.unwrap();
        //~^ ERROR: this call to `unwrap()` will always panic
        // will panic
        x.expect("an error message");
        //~^ ERROR: this call to `expect()` will always panic
        // unnecessary
        x.unwrap_err();
        //~^ ERROR: called `unwrap_err` on `x` after checking its variant with `is_ok`
    }
    if x.is_err() {
        // will panic
        x.unwrap();
        //~^ ERROR: this call to `unwrap()` will always panic
        // unnecessary
        x.unwrap_err();
        //~^ ERROR: called `unwrap_err` on `x` after checking its variant with `is_err`
    } else {
        // unnecessary
        x.unwrap();
        //~^ ERROR: called `unwrap` on `x` after checking its variant with `is_err`
        // will panic
        x.unwrap_err();
        //~^ ERROR: this call to `unwrap_err()` will always panic
    }
    if x.is_ok() {
        x = Err(());
        // not unnecessary because of mutation of x
        // it will always panic but the lint is not smart enough to see this (it only
        // checks if conditions).
        x.unwrap();
    } else {
        x = Ok(());
        // not unnecessary because of mutation of x
        // it will always panic but the lint is not smart enough to see this (it
        // only checks if conditions).
        x.unwrap_err();
    }

    // ok, it's a common test pattern
    assert!(x.is_ok(), "{:?}", x.unwrap_err());
}

fn issue11371() {
    let option = Some(());

    if option.is_some() {
        option.as_ref().unwrap();
        //~^ ERROR: called `unwrap` on `option` after checking its variant with `is_some`
    } else {
        option.as_ref().unwrap();
        //~^ ERROR: this call to `unwrap()` will always panic
    }

    let result = Ok::<(), ()>(());

    if result.is_ok() {
        result.as_ref().unwrap();
        //~^ ERROR: called `unwrap` on `result` after checking its variant with `is_ok`
    } else {
        result.as_ref().unwrap();
        //~^ ERROR: this call to `unwrap()` will always panic
    }

    let mut option = Some(());
    if option.is_some() {
        option.as_mut().unwrap();
        //~^ ERROR: called `unwrap` on `option` after checking its variant with `is_some`
    } else {
        option.as_mut().unwrap();
        //~^ ERROR: this call to `unwrap()` will always panic
    }

    let mut result = Ok::<(), ()>(());
    if result.is_ok() {
        result.as_mut().unwrap();
        //~^ ERROR: called `unwrap` on `result` after checking its variant with `is_ok`
    } else {
        result.as_mut().unwrap();
        //~^ ERROR: this call to `unwrap()` will always panic
    }

    // This should not lint. Statics are, at the time of writing, not linted on anyway,
    // but if at some point they are supported by this lint, it should correctly see that
    // `X` is being mutated and not suggest `if let Some(..) = X {}`
    static mut X: Option<i32> = Some(123);
    unsafe {
        if X.is_some() {
            X = None;
            X.unwrap();
        }
    }
}

fn check_expect() {
    let x = Some(());
    if x.is_some() {
        #[expect(clippy::unnecessary_unwrap)]
        // unnecessary
        x.unwrap();
        #[expect(clippy::unnecessary_unwrap)]
        // unnecessary
        x.expect("an error message");
    } else {
        #[expect(clippy::panicking_unwrap)]
        // will panic
        x.unwrap();
        #[expect(clippy::panicking_unwrap)]
        // will panic
        x.expect("an error message");
    }
}
