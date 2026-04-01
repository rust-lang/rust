#![warn(clippy::manual_map)]
#![allow(clippy::toplevel_ref_arg)]

fn main() {
    // Lint. `y` is declared within the arm, so it isn't captured by the map closure
    let _ = match Some(0) {
        //~^ manual_map
        Some(x) => Some({
            let y = (String::new(), String::new());
            (x, y.0)
        }),
        None => None,
    };

    // Don't lint. `s` is borrowed until partway through the arm, but needs to be captured by the map
    // closure
    let s = Some(String::new());
    let _ = match &s {
        Some(x) => Some((x.clone(), s)),
        None => None,
    };

    // Don't lint. `s` is borrowed until partway through the arm, but needs to be captured by the map
    // closure
    let s = Some(String::new());
    let _ = match &s {
        Some(x) => Some({
            let clone = x.clone();
            let s = || s;
            (clone, s())
        }),
        None => None,
    };

    // Don't lint. `s` is borrowed until partway through the arm, but needs to be captured as a mutable
    // reference by the map closure
    let mut s = Some(String::new());
    let _ = match &s {
        Some(x) => Some({
            let clone = x.clone();
            let ref mut s = s;
            (clone, s)
        }),
        None => None,
    };

    let s = Some(String::new());
    // Lint. `s` is captured by reference, so no lifetime issues.
    let _ = match &s {
        //~^ manual_map
        Some(x) => Some({ if let Some(ref s) = s { (x.clone(), s) } else { panic!() } }),
        None => None,
    };
    // Don't lint this, type of `s` is coercioned from `&String` to `&str`
    let x: Option<(String, &str)> = match &s {
        Some(x) => Some({ if let Some(ref s) = s { (x.clone(), s) } else { panic!() } }),
        None => None,
    };

    // Issue #7820
    unsafe fn f(x: u32) -> u32 {
        x
    }
    unsafe {
        let _ = match Some(0) {
            //~^ manual_map
            Some(x) => Some(f(x)),
            None => None,
        };
    }
    let _ = match Some(0) {
        //~^ manual_map
        Some(x) => unsafe { Some(f(x)) },
        None => None,
    };
    let _ = match Some(0) {
        //~^ manual_map
        Some(x) => Some(unsafe { f(x) }),
        None => None,
    };
}

// issue #12659
mod with_type_coercion {
    trait DummyTrait {}

    fn foo<T: DummyTrait, F: Fn() -> Result<T, ()>>(f: F) {
        // Don't lint
        let _: Option<Result<Box<dyn DummyTrait>, ()>> = match Some(0) {
            Some(_) => Some(match f() {
                Ok(res) => Ok(Box::new(res)),
                _ => Err(()),
            }),
            None => None,
        };

        let _: Option<Box<&[u8]>> = match Some(()) {
            Some(_) => Some(Box::new(b"1234")),
            None => None,
        };

        let x = String::new();
        let _: Option<Box<&str>> = match Some(()) {
            Some(_) => Some(Box::new(&x)),
            None => None,
        };

        let _: Option<&str> = match Some(()) {
            Some(_) => Some(&x),
            None => None,
        };

        let _ = match Some(0) {
            //~^ manual_map
            Some(_) => Some(match f() {
                Ok(res) => Ok(Box::new(res)),
                _ => Err(()),
            }),
            None => None,
        };
    }

    #[allow(clippy::redundant_allocation)]
    fn bar() {
        fn f(_: Option<Box<&[u8]>>) {}
        fn g(b: &[u8]) -> Box<&[u8]> {
            Box::new(b)
        }

        let x: &[u8; 4] = b"1234";
        f(match Some(()) {
            Some(_) => Some(Box::new(x)),
            None => None,
        });

        let _: Option<Box<&[u8]>> = match Some(0) {
            //~^ manual_map
            Some(_) => Some(g(x)),
            None => None,
        };
    }

    fn with_fn_ret(s: &Option<String>) -> Option<(String, &str)> {
        // Don't lint, `map` doesn't work as the return type is adjusted.
        match s {
            Some(x) => Some({ if let Some(s) = s { (x.clone(), s) } else { panic!() } }),
            None => None,
        }
    }

    fn with_fn_ret_2(s: &Option<String>) -> Option<(String, &str)> {
        if true {
            // Don't lint, `map` doesn't work as the return type is adjusted.
            return match s {
                Some(x) => Some({ if let Some(s) = s { (x.clone(), s) } else { panic!() } }),
                None => None,
            };
        }
        None
    }

    #[allow(clippy::needless_late_init)]
    fn with_fn_ret_3<'a>(s: &'a Option<String>) -> Option<(String, &'a str)> {
        let x: Option<(String, &'a str)>;
        x = {
            match s {
                Some(x) => Some({ if let Some(s) = s { (x.clone(), s) } else { panic!() } }),
                None => None,
            }
        };
        x
    }
}
