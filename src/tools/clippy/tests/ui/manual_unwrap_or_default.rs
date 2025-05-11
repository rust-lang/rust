#![warn(clippy::manual_unwrap_or_default)]
#![allow(clippy::unnecessary_literal_unwrap)]

fn main() {
    let x: Option<Vec<String>> = None;
    match x {
        //~^ manual_unwrap_or_default
        Some(v) => v,
        None => Vec::default(),
    };

    let x: Option<Vec<String>> = None;
    match x {
        //~^ manual_unwrap_or_default
        Some(v) => v,
        _ => Vec::default(),
    };

    let x: Option<String> = None;
    match x {
        //~^ manual_unwrap_or_default
        Some(v) => v,
        None => String::new(),
    };

    let x: Option<Vec<String>> = None;
    match x {
        //~^ manual_unwrap_or_default
        None => Vec::default(),
        Some(v) => v,
    };

    let x: Option<Vec<String>> = None;
    if let Some(v) = x {
        //~^ manual_unwrap_or_default

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

    let x: Result<String, i64> = Ok(String::new());
    match x {
        //~^ manual_unwrap_or_default
        Ok(v) => v,
        Err(_) => String::new(),
    };

    let x: Result<String, i64> = Ok(String::new());
    if let Ok(v) = x {
        //~^ manual_unwrap_or_default

        v
    } else {
        String::new()
    };
}

// Issue #12531
unsafe fn no_deref_ptr(a: Option<i32>, b: *const Option<i32>) -> i32 {
    unsafe {
        match a {
            // `*b` being correct depends on `a == Some(_)`
            Some(_) => match *b {
                //~^ manual_unwrap_or_default
                Some(v) => v,
                _ => 0,
            },
            _ => 0,
        }
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

// Should not warn!
fn issue_12928() {
    let x = Some((1, 2));
    let y = if let Some((a, _)) = x { a } else { 0 };
    let y = if let Some((a, ..)) = x { a } else { 0 };
    let x = Some([1, 2]);
    let y = if let Some([a, _]) = x { a } else { 0 };
    let y = if let Some([a, ..]) = x { a } else { 0 };

    struct X {
        a: u8,
        b: u8,
    }
    let x = Some(X { a: 0, b: 0 });
    let y = if let Some(X { a, .. }) = x { a } else { 0 };
    struct Y(u8, u8);
    let x = Some(Y(0, 0));
    let y = if let Some(Y(a, _)) = x { a } else { 0 };
    let y = if let Some(Y(a, ..)) = x { a } else { 0 };
}

// For symetry with `manual_unwrap_or` test
fn allowed_manual_unwrap_or_zero() -> u32 {
    if let Some(x) = Some(42) {
        //~^ manual_unwrap_or_default
        x
    } else {
        0
    }
}
