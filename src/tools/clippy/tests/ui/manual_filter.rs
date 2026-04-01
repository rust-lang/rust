#![warn(clippy::manual_filter)]
#![allow(unused_variables, dead_code, clippy::useless_vec)]

fn main() {
    match Some(0) {
        //~^ manual_filter
        None => None,
        Some(x) => {
            if x > 0 {
                None
            } else {
                Some(x)
            }
        },
    };

    match Some(1) {
        //~^ manual_filter
        Some(x) => {
            if x > 0 {
                None
            } else {
                Some(x)
            }
        },
        None => None,
    };

    match Some(2) {
        //~^ manual_filter
        Some(x) => {
            if x > 0 {
                None
            } else {
                Some(x)
            }
        },
        _ => None,
    };

    match Some(3) {
        //~^ manual_filter
        Some(x) => {
            if x > 0 {
                Some(x)
            } else {
                None
            }
        },
        None => None,
    };

    let y = Some(4);
    match y {
        //~^ manual_filter
        // Some(4)
        None => None,
        Some(x) => {
            if x > 0 {
                None
            } else {
                Some(x)
            }
        },
    };

    match Some(5) {
        //~^ manual_filter
        Some(x) => {
            if x > 0 {
                Some(x)
            } else {
                None
            }
        },
        _ => None,
    };

    match Some(6) {
        //~^ manual_filter
        Some(ref x) => {
            if x > &0 {
                Some(x)
            } else {
                None
            }
        },
        _ => None,
    };

    let external_cond = true;
    match Some(String::new()) {
        //~^ manual_filter
        Some(x) => {
            if external_cond {
                Some(x)
            } else {
                None
            }
        },
        _ => None,
    };

    if let Some(x) = Some(7) {
        //~^ manual_filter
        if external_cond { Some(x) } else { None }
    } else {
        None
    };

    match &Some(8) {
        //~^ manual_filter
        &Some(x) => {
            if x != 0 {
                Some(x)
            } else {
                None
            }
        },
        _ => None,
    };

    match Some(9) {
        //~^ manual_filter
        Some(x) => {
            if x > 10 && x < 100 {
                Some(x)
            } else {
                None
            }
        },
        None => None,
    };

    const fn f1() {
        // Don't lint, `.filter` is not const
        match Some(10) {
            Some(x) => {
                if x > 10 && x < 100 {
                    Some(x)
                } else {
                    None
                }
            },
            None => None,
        };
    }

    #[allow(clippy::blocks_in_conditions)]
    match Some(11) {
        //~^ manual_filter
        // Lint, statement is preserved by `.filter`
        Some(x) => {
            if {
                println!("foo");
                x > 10 && x < 100
            } {
                Some(x)
            } else {
                None
            }
        },
        None => None,
    };

    match Some(12) {
        // Don't Lint, statement is lost by `.filter`
        Some(x) => {
            if x > 10 && x < 100 {
                println!("foo");
                Some(x)
            } else {
                None
            }
        },
        None => None,
    };

    match Some(13) {
        // Don't Lint, because of `None => Some(1)`
        Some(x) => {
            if x > 10 && x < 100 {
                println!("foo");
                Some(x)
            } else {
                None
            }
        },
        None => Some(1),
    };

    unsafe fn f(x: u32) -> bool {
        true
    }
    let _ = match Some(14) {
        //~^ manual_filter
        Some(x) => {
            if unsafe { f(x) } {
                Some(x)
            } else {
                None
            }
        },
        None => None,
    };
    let _ = match Some(15) {
        //~^ manual_filter
        Some(x) => unsafe { if f(x) { Some(x) } else { None } },
        None => None,
    };

    #[allow(clippy::redundant_pattern_matching)]
    if let Some(_) = Some(16) {
        Some(16)
    } else if let Some(x) = Some(16) {
        //~^ manual_filter
        // Lint starting from here
        if x % 2 == 0 { Some(x) } else { None }
    } else {
        None
    };

    match Some((17, 17)) {
        // Not linted for now could be
        Some((x, y)) => {
            if y != x {
                Some((x, y))
            } else {
                None
            }
        },
        None => None,
    };

    struct NamedTuple {
        pub x: u8,
        pub y: (i32, u32),
    }

    match Some(NamedTuple {
        // Not linted for now could be
        x: 17,
        y: (18, 19),
    }) {
        Some(NamedTuple { x, y }) => {
            if y.1 != x as u32 {
                Some(NamedTuple { x, y })
            } else {
                None
            }
        },
        None => None,
    };

    match Some(20) {
        // Don't Lint, because `Some(3*x)` is not `None`
        None => None,
        Some(x) => {
            if x > 0 {
                Some(3 * x)
            } else {
                Some(x)
            }
        },
    };

    // Don't lint: https://github.com/rust-lang/rust-clippy/issues/10088
    let result = if let Some(a) = maybe_some() {
        if let Some(b) = maybe_some() {
            Some(a + b)
        } else {
            Some(a)
        }
    } else {
        None
    };

    let allowed_integers = vec![3, 4, 5, 6];
    // Don't lint, since there's a side effect in the else branch
    match Some(21) {
        Some(x) => {
            if allowed_integers.contains(&x) {
                Some(x)
            } else {
                println!("Invalid integer: {x:?}");
                None
            }
        },
        None => None,
    };
}

fn maybe_some() -> Option<u32> {
    Some(0)
}
