// run-rustfix
#![warn(clippy::option_if_let_else)]

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

fn unop_bad(string: &Option<&str>) -> usize {
    if let Some(s) = *string {
        s.len()
    } else {
        0
    }
}

fn longer_body(arg: Option<u32>) -> u32 {
    if let Some(x) = arg {
        let y = x * x;
        y * y
    } else {
        13
    }
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

fn main() {
    let optional = Some(5);
    let _ = if let Some(x) = optional { x + 2 } else { 5 };
    let _ = bad1(None);
    let _ = else_if_option(None);
    let _ = unop_bad(&None);
    let _ = longer_body(None);
    test_map_or_else(None);
    let _ = negative_tests(None);
}
