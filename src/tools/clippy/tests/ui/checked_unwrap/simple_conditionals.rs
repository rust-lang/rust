//@no-rustfix: overlapping suggestions
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
            //~^ unnecessary_unwrap
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
        //~^ unnecessary_unwrap

        // unnecessary
        x.expect("an error message");
        //~^ unnecessary_unwrap
    } else {
        // will panic
        x.unwrap();
        //~^ panicking_unwrap

        // will panic
        x.expect("an error message");
        //~^ panicking_unwrap
    }
    if x.is_none() {
        // will panic
        x.unwrap();
        //~^ panicking_unwrap
    } else {
        // unnecessary
        x.unwrap();
        //~^ unnecessary_unwrap
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
        //~^ unnecessary_unwrap

        // unnecessary
        x.expect("an error message");
        //~^ unnecessary_unwrap

        // will panic
        x.unwrap_err();
        //~^ panicking_unwrap
    } else {
        // will panic
        x.unwrap();
        //~^ panicking_unwrap

        // will panic
        x.expect("an error message");
        //~^ panicking_unwrap

        // unnecessary
        x.unwrap_err();
        //~^ unnecessary_unwrap
    }
    if x.is_err() {
        // will panic
        x.unwrap();
        //~^ panicking_unwrap

        // unnecessary
        x.unwrap_err();
        //~^ unnecessary_unwrap
    } else {
        // unnecessary
        x.unwrap();
        //~^ unnecessary_unwrap

        // will panic
        x.unwrap_err();
        //~^ panicking_unwrap
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
        //~^ unnecessary_unwrap
    } else {
        option.as_ref().unwrap();
        //~^ panicking_unwrap
    }

    let result = Ok::<(), ()>(());

    if result.is_ok() {
        result.as_ref().unwrap();
        //~^ unnecessary_unwrap
    } else {
        result.as_ref().unwrap();
        //~^ panicking_unwrap
    }

    let mut option = Some(());
    if option.is_some() {
        option.as_mut().unwrap();
        //~^ unnecessary_unwrap
    } else {
        option.as_mut().unwrap();
        //~^ panicking_unwrap
    }

    let mut result = Ok::<(), ()>(());
    if result.is_ok() {
        result.as_mut().unwrap();
        //~^ unnecessary_unwrap
    } else {
        result.as_mut().unwrap();
        //~^ panicking_unwrap
    }

    // This should not lint. Statics are, at the time of writing, not linted on anyway,
    // but if at some point they are supported by this lint, it should correctly see that
    // `X` is being mutated and not suggest `if let Some(..) = X {}`
    static mut X: Option<i32> = Some(123);
    unsafe {
        if X.is_some() {
            //~^ ERROR: creating a shared reference
            X = None;
            X.unwrap();
        }
    }
}

fn gen_option() -> Option<()> {
    Some(())
    // Or None
}

fn gen_result() -> Result<(), ()> {
    Ok(())
    // Or Err(())
}

fn issue14725() {
    let option = Some(());

    if option.is_some() {
        let _ = option.as_ref().unwrap();
        //~^ unnecessary_unwrap
    } else {
        let _ = option.as_ref().unwrap();
        //~^ panicking_unwrap
    }

    let result = Ok::<(), ()>(());

    if result.is_ok() {
        let _y = 1;
        result.as_ref().unwrap();
        //~^ unnecessary_unwrap
    } else {
        let _y = 1;
        result.as_ref().unwrap();
        //~^ panicking_unwrap
    }

    let mut option = Some(());
    if option.is_some() {
        option = gen_option();
        option.as_mut().unwrap();
    } else {
        option = gen_option();
        option.as_mut().unwrap();
    }

    let mut result = Ok::<(), ()>(());
    if result.is_ok() {
        result = gen_result();
        result.as_mut().unwrap();
    } else {
        result = gen_result();
        result.as_mut().unwrap();
    }
}

fn issue14763(x: Option<String>, r: Result<(), ()>) {
    _ = || {
        if x.is_some() {
            _ = x.unwrap();
            //~^ unnecessary_unwrap
        } else {
            _ = x.unwrap();
            //~^ panicking_unwrap
        }
    };
    _ = || {
        if r.is_ok() {
            _ = r.as_ref().unwrap();
            //~^ unnecessary_unwrap
        } else {
            _ = r.as_ref().unwrap();
            //~^ panicking_unwrap
        }
    };
}

const ISSUE14763: fn(Option<String>) = |x| {
    _ = || {
        if x.is_some() {
            _ = x.unwrap();
            //~^ unnecessary_unwrap
        } else {
            _ = x.unwrap();
            //~^ panicking_unwrap
        }
    }
};

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
