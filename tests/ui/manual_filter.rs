// run-rustfix

#![warn(clippy::manual_filter)]
#![allow(unused_variables, dead_code)]

fn main() {
    match Some(0) {
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
        if external_cond { Some(x) } else { None }
    } else {
        None
    };

    match &Some(8) {
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

    #[allow(clippy::blocks_in_if_conditions)]
    match Some(11) {
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
        Some(x) => unsafe {
            if f(x) { Some(x) } else { None }
        },
        None => None,
    };

    #[allow(clippy::redundant_pattern_matching)]
    if let Some(_) = Some(16) {
        Some(16)
    } else if let Some(x) = Some(16) {
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
}

fn maybe_some() -> Option<u32> {
    Some(0)
}
