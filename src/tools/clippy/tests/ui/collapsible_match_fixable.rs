#![warn(clippy::collapsible_match)]
#![allow(clippy::single_match, clippy::redundant_guards)]

fn issue16558() {
    let opt = Some(1);
    let _ = match opt {
        Some(s) => {
            if s == 1 { s } else { 1 }
            //~^ collapsible_match
        },
        _ => 1,
    };

    match opt {
        Some(s) => {
            (if s == 1 {
                //~^ collapsible_match
                todo!()
            })
        },
        _ => {},
    };

    let _ = match opt {
        Some(s) if s > 2 => {
            if s == 1 { s } else { 1 }
            //~^ collapsible_match
        },
        _ => 1,
    };
}

// https://github.com/rust-lang/rust-clippy/issues/16875
// lint still fires when only wildcard-like arms follow (fall-through is harmless)
fn issue16875(a: Option<&str>, b: i32) -> i32 {
    let mut res = 0;
    match a {
        Some(_) => {
            if b == 0 {
                //~^ collapsible_match
                res = 1;
            }
        },
        _ => {},
    }
    res
}
