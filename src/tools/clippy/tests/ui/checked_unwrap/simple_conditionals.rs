//@no-rustfix: has placeholders
#![warn(clippy::panicking_unwrap, clippy::unnecessary_unwrap)]
#![expect(
    clippy::if_same_then_else,
    clippy::branches_sharing_code,
    clippy::unnecessary_literal_unwrap,
    clippy::self_assignment
)]

macro_rules! m {
    ($a:expr) => {
        if $a.is_some() {
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
        x.unwrap();
        //~^ unnecessary_unwrap

        x.expect("an error message");
        //~^ unnecessary_unwrap
    } else {
        x.unwrap();
        //~^ panicking_unwrap

        x.expect("an error message");
        //~^ panicking_unwrap
    }
    if x.is_none() {
        x.unwrap();
        //~^ panicking_unwrap
    } else {
        x.unwrap();
        //~^ unnecessary_unwrap
    }
    m!(x);
    checks_in_param!(x.is_some(), x.unwrap());
    checks_unwrap!(x, x.unwrap());
    checks_some!(x.is_some(), x);
    let mut x: Result<(), ()> = Ok(());
    if x.is_ok() {
        x.unwrap();
        //~^ unnecessary_unwrap

        x.expect("an error message");
        //~^ unnecessary_unwrap

        x.unwrap_err();
        //~^ panicking_unwrap
    } else {
        x.unwrap();
        //~^ panicking_unwrap

        x.expect("an error message");
        //~^ panicking_unwrap

        x.unwrap_err();
        //~^ unnecessary_unwrap
    }
    if x.is_err() {
        x.unwrap();
        //~^ panicking_unwrap

        x.unwrap_err();
        //~^ unnecessary_unwrap
    } else {
        x.unwrap();
        //~^ unnecessary_unwrap

        x.unwrap_err();
        //~^ panicking_unwrap
    }
    if x.is_ok() {
        x = Err(());
        // not unnecessary because of mutation of `x`
        // it will always panic but the lint is not smart enough to see this (it only
        // checks if conditions).
        x.unwrap();
    } else {
        x = Ok(());
        // not unnecessary because of mutation of `x`
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

    // This should not lint and suggest `if let Some(..) = X {}`, as `X` is being mutated
    static mut X: Option<i32> = Some(123);
    unsafe {
        #[expect(static_mut_refs)]
        if X.is_some() {
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

fn issue12295() {
    let option = Some(());

    if option.is_some() {
        println!("{:?}", option.unwrap());
        //~^ unnecessary_unwrap
    } else {
        println!("{:?}", option.unwrap());
        //~^ panicking_unwrap
    }

    let result = Ok::<(), ()>(());

    if result.is_ok() {
        println!("{:?}", result.unwrap());
        //~^ unnecessary_unwrap
    } else {
        println!("{:?}", result.unwrap());
        //~^ panicking_unwrap
    }
}

fn check_expect() {
    let x = Some(());
    if x.is_some() {
        #[expect(clippy::unnecessary_unwrap)]
        x.unwrap();
        #[expect(clippy::unnecessary_unwrap)]
        x.expect("an error message");
    } else {
        #[expect(clippy::panicking_unwrap)]
        x.unwrap();
        #[expect(clippy::panicking_unwrap)]
        x.expect("an error message");
    }
}

fn partial_moves() {
    fn borrow_option(_: &Option<()>) {}

    let x = Some(());
    // Using `if let Some(o) = x` won't work here, as `borrow_option` will try to borrow a moved value
    if x.is_some() {
        borrow_option(&x);
        x.unwrap();
        //~^ unnecessary_unwrap
    }
    // This is fine though, as `if let Some(o) = &x` won't move `x`
    if x.is_some() {
        borrow_option(&x);
        x.as_ref().unwrap();
        //~^ unnecessary_unwrap
    }
}

fn issue15321() {
    struct Soption {
        option: Option<bool>,
        other: bool,
    }
    let mut sopt = Soption {
        option: Some(true),
        other: true,
    };
    // Lint: nothing was mutated
    let _res = if sopt.option.is_some() {
        sopt.option.unwrap()
        //~^ unnecessary_unwrap
    } else {
        sopt.option.unwrap()
        //~^ panicking_unwrap
    };
    // Lint: an unrelated field was mutated
    let _res = if sopt.option.is_some() {
        sopt.other = false;
        sopt.option.unwrap()
        //~^ unnecessary_unwrap
    } else {
        sopt.other = false;
        sopt.option.unwrap()
        //~^ panicking_unwrap
    };
    // No lint: the whole local was mutated
    let _res = if sopt.option.is_some() {
        sopt = sopt;
        sopt.option.unwrap()
    } else {
        sopt.option = None;
        sopt.option.unwrap()
    };
    // No lint: the field we're looking at was mutated
    let _res = if sopt.option.is_some() {
        sopt = sopt;
        sopt.option.unwrap()
    } else {
        sopt.option = None;
        sopt.option.unwrap()
    };

    struct Toption(Option<bool>, bool);
    let mut topt = Toption(Some(true), true);
    // Lint: nothing was mutated
    let _res = if topt.0.is_some() {
        topt.0.unwrap()
        //~^ unnecessary_unwrap
    } else {
        topt.0.unwrap()
        //~^ panicking_unwrap
    };
    // Lint: an unrelated field was mutated
    let _res = if topt.0.is_some() {
        topt.1 = false;
        topt.0.unwrap()
        //~^ unnecessary_unwrap
    } else {
        topt.1 = false;
        topt.0.unwrap()
        //~^ panicking_unwrap
    };
    // No lint: the whole local was mutated
    let _res = if topt.0.is_some() {
        topt = topt;
        topt.0.unwrap()
    } else {
        topt = topt;
        topt.0.unwrap()
    };
    // No lint: the field we're looking at was mutated
    let _res = if topt.0.is_some() {
        topt.0 = None;
        topt.0.unwrap()
    } else {
        topt.0 = None;
        topt.0.unwrap()
    };

    // Nested field accesses get linted as well
    struct Soption2 {
        other: bool,
        option: Soption,
    }
    let mut sopt2 = Soption2 {
        other: true,
        option: Soption {
            option: Some(true),
            other: true,
        },
    };
    // Lint: no fields were mutated
    let _res = if sopt2.option.option.is_some() {
        sopt2.option.option.unwrap()
        //~^ unnecessary_unwrap
    } else {
        sopt2.option.option.unwrap()
        //~^ panicking_unwrap
    };
    // Lint: an unrelated outer field was mutated -- don't get confused by `Soption2.other` having the
    // same `FieldIdx` of 1 as `Soption.option`
    let _res = if sopt2.option.option.is_some() {
        sopt2.other = false;
        sopt2.option.option.unwrap()
        //~^ unnecessary_unwrap
    } else {
        sopt2.other = false;
        sopt2.option.option.unwrap()
        //~^ panicking_unwrap
    };
    // Lint: an unrelated inner field was mutated
    let _res = if sopt2.option.option.is_some() {
        sopt2.option.other = false;
        sopt2.option.option.unwrap()
        //~^ unnecessary_unwrap
    } else {
        sopt2.option.other = false;
        sopt2.option.option.unwrap()
        //~^ panicking_unwrap
    };
    // Don't lint: the whole local was mutated
    let _res = if sopt2.option.option.is_some() {
        sopt2 = sopt2;
        sopt2.option.option.unwrap()
    } else {
        sopt2 = sopt2;
        sopt2.option.option.unwrap()
    };
    // Don't lint: a parent field of the field we're looking at was mutated, and with that the
    // field we're looking at
    let _res = if sopt2.option.option.is_some() {
        sopt2.option = sopt;
        sopt2.option.option.unwrap()
    } else {
        sopt2.option = sopt;
        sopt2.option.option.unwrap()
    };
    // Don't lint: the field we're looking at was mutated directly
    let _res = if sopt2.option.option.is_some() {
        sopt2.option.option = None;
        sopt2.option.option.unwrap()
    } else {
        sopt2.option.option = None;
        sopt2.option.option.unwrap()
    };

    // Partial moves
    fn borrow_toption(_: &Toption) {}

    // Using `if let Some(o) = topt.0` won't work here, as `borrow_toption` will try to borrow a
    // partially moved value
    if topt.0.is_some() {
        borrow_toption(&topt);
        topt.0.unwrap();
        //~^ unnecessary_unwrap
    }
    // This is fine though, as `if let Some(o) = &topt.0` won't (partially) move `topt`
    if topt.0.is_some() {
        borrow_toption(&topt);
        topt.0.as_ref().unwrap();
        //~^ unnecessary_unwrap
    }
}

mod issue16188 {
    struct Foo {
        value: Option<i32>,
    }

    impl Foo {
        pub fn bar(&mut self) {
            let print_value = |v: i32| {
                println!("{}", v);
            };

            if self.value.is_none() {
                self.value = Some(10);
                print_value(self.value.unwrap());
            }
        }
    }
}
