#![warn(clippy::manual_unwrap_or_default)]
#![allow(clippy::unnecessary_literal_unwrap)]

fn main() {
    let x: Option<Vec<String>> = None;
    match x {
        //~^ ERROR: match can be simplified with `.unwrap_or_default()`
        Some(v) => v,
        None => Vec::default(),
    };

    let x: Option<Vec<String>> = None;
    match x {
        //~^ ERROR: match can be simplified with `.unwrap_or_default()`
        Some(v) => v,
        _ => Vec::default(),
    };

    let x: Option<String> = None;
    match x {
        //~^ ERROR: match can be simplified with `.unwrap_or_default()`
        Some(v) => v,
        None => String::new(),
    };

    let x: Option<Vec<String>> = None;
    match x {
        //~^ ERROR: match can be simplified with `.unwrap_or_default()`
        None => Vec::default(),
        Some(v) => v,
    };

    let x: Option<Vec<String>> = None;
    if let Some(v) = x {
        //~^ ERROR: if let can be simplified with `.unwrap_or_default()`
        v
    } else {
        Vec::default()
    };
}

// Issue #12531
unsafe fn no_deref_ptr(a: Option<i32>, b: *const Option<i32>) -> i32 {
    match a {
        // `*b` being correct depends on `a == Some(_)`
        Some(_) => match *b {
            Some(v) => v,
            _ => 0,
        },
        _ => 0,
    }
}
