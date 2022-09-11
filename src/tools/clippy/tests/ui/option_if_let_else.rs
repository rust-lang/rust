// run-rustfix
#![warn(clippy::option_if_let_else)]
#![allow(
    unused_tuple_struct_fields,
    clippy::redundant_closure,
    clippy::ref_option_ref,
    clippy::equatable_if_let,
    clippy::let_unit_value
)]

fn bad1(string: Option<&str>) -> (bool, &str) {
    if let Some(x) = string {
        (true, x)
    } else {
        (false, "hello")
    }
}

fn else_if_option(string: Option<&str>) -> Option<(bool, &str)> {
    if string.is_none() {
        None
    } else if let Some(x) = string {
        Some((true, x))
    } else {
        Some((false, ""))
    }
}

fn unop_bad(string: &Option<&str>, mut num: Option<i32>) {
    let _ = if let Some(s) = *string { s.len() } else { 0 };
    let _ = if let Some(s) = &num { s } else { &0 };
    let _ = if let Some(s) = &mut num {
        *s += 1;
        s
    } else {
        &mut 0
    };
    let _ = if let Some(ref s) = num { s } else { &0 };
    let _ = if let Some(mut s) = num {
        s += 1;
        s
    } else {
        0
    };
    let _ = if let Some(ref mut s) = num {
        *s += 1;
        s
    } else {
        &mut 0
    };
}

fn longer_body(arg: Option<u32>) -> u32 {
    if let Some(x) = arg {
        let y = x * x;
        y * y
    } else {
        13
    }
}

fn impure_else(arg: Option<i32>) {
    let side_effect = || {
        println!("return 1");
        1
    };
    let _ = if let Some(x) = arg {
        x
    } else {
        // map_or_else must be suggested
        side_effect()
    };
}

fn test_map_or_else(arg: Option<u32>) {
    let _ = if let Some(x) = arg {
        x * x * x * x
    } else {
        let mut y = 1;
        y = (y + 2 / y) / 2;
        y = (y + 2 / y) / 2;
        y
    };
}

fn negative_tests(arg: Option<u32>) -> u32 {
    let _ = if let Some(13) = arg { "unlucky" } else { "lucky" };
    for _ in 0..10 {
        let _ = if let Some(x) = arg {
            x
        } else {
            continue;
        };
    }
    let _ = if let Some(x) = arg {
        return x;
    } else {
        5
    };
    7
}

// #7973
fn pattern_to_vec(pattern: &str) -> Vec<String> {
    pattern
        .trim_matches('/')
        .split('/')
        .flat_map(|s| {
            if let Some(idx) = s.find('.') {
                vec![s[..idx].to_string(), s[idx..].to_string()]
            } else {
                vec![s.to_string()]
            }
        })
        .collect::<Vec<_>>()
}

enum DummyEnum {
    One(u8),
    Two,
}

// should not warn since there is a compled complex subpat
// see #7991
fn complex_subpat() -> DummyEnum {
    let x = Some(DummyEnum::One(1));
    let _ = if let Some(_one @ DummyEnum::One(..)) = x { 1 } else { 2 };
    DummyEnum::Two
}

fn main() {
    let optional = Some(5);
    let _ = if let Some(x) = optional { x + 2 } else { 5 };
    let _ = bad1(None);
    let _ = else_if_option(None);
    unop_bad(&None, None);
    let _ = longer_body(None);
    test_map_or_else(None);
    let _ = negative_tests(None);
    let _ = impure_else(None);

    let _ = if let Some(x) = Some(0) {
        loop {
            if x == 0 {
                break x;
            }
        }
    } else {
        0
    };

    // #7576
    const fn _f(x: Option<u32>) -> u32 {
        // Don't lint, `map_or` isn't const
        if let Some(x) = x { x } else { 10 }
    }

    // #5822
    let s = String::new();
    // Don't lint, `Some` branch consumes `s`, but else branch uses `s`
    let _ = if let Some(x) = Some(0) {
        let s = s;
        s.len() + x
    } else {
        s.len()
    };

    let s = String::new();
    // Lint, both branches immutably borrow `s`.
    let _ = if let Some(x) = Some(0) { s.len() + x } else { s.len() };

    let s = String::new();
    // Lint, `Some` branch consumes `s`, but else branch doesn't use `s`.
    let _ = if let Some(x) = Some(0) {
        let s = s;
        s.len() + x
    } else {
        1
    };

    let s = Some(String::new());
    // Don't lint, `Some` branch borrows `s`, but else branch consumes `s`
    let _ = if let Some(x) = &s {
        x.len()
    } else {
        let _s = s;
        10
    };

    let mut s = Some(String::new());
    // Don't lint, `Some` branch mutably borrows `s`, but else branch also borrows  `s`
    let _ = if let Some(x) = &mut s {
        x.push_str("test");
        x.len()
    } else {
        let _s = &s;
        10
    };

    async fn _f1(x: u32) -> u32 {
        x
    }

    async fn _f2() {
        // Don't lint. `await` can't be moved into a closure.
        let _ = if let Some(x) = Some(0) { _f1(x).await } else { 0 };
    }

    let _ = pattern_to_vec("hello world");
    let _ = complex_subpat();

    // issue #8492
    let _ = match s {
        Some(string) => string.len(),
        None => 1,
    };
    let _ = match Some(10) {
        Some(a) => a + 1,
        None => 5,
    };

    let res: Result<i32, i32> = Ok(5);
    let _ = match res {
        Ok(a) => a + 1,
        _ => 1,
    };
    let _ = match res {
        Err(_) => 1,
        Ok(a) => a + 1,
    };
    let _ = if let Ok(a) = res { a + 1 } else { 5 };
}
