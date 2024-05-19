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

    // Issue #12564
    // No error as &Vec<_> doesn't implement std::default::Default
    let mut map = std::collections::HashMap::from([(0, vec![0; 3]), (1, vec![1; 3]), (2, vec![2])]);
    let x: &[_] = if let Some(x) = map.get(&0) { x } else { &[] };
    // Same code as above written using match.
    let x: &[_] = match map.get(&0) {
        Some(x) => x,
        None => &[],
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

const fn issue_12568(opt: Option<bool>) -> bool {
    match opt {
        Some(s) => s,
        None => false,
    }
}

fn issue_12569() {
    let match_none_se = match 1u32.checked_div(0) {
        Some(v) => v,
        None => {
            println!("important");
            0
        },
    };
    let match_some_se = match 1u32.checked_div(0) {
        Some(v) => {
            println!("important");
            v
        },
        None => 0,
    };
    let iflet_else_se = if let Some(v) = 1u32.checked_div(0) {
        v
    } else {
        println!("important");
        0
    };
    let iflet_then_se = if let Some(v) = 1u32.checked_div(0) {
        println!("important");
        v
    } else {
        0
    };
}
