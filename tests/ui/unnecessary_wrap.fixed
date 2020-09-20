// run-rustfix

#![warn(clippy::unnecessary_wrap)]
#![allow(clippy::no_effect)]
#![allow(clippy::needless_return)]
#![allow(clippy::if_same_then_else)]
#![allow(dead_code)]

// should be linted
fn func1(a: bool, b: bool) -> Option<i32> {
    if a && b {
        return Some(42);
    }
    if a {
        Some(-1);
        Some(2)
    } else {
        return Some(1337);
    }
}

// public fns should not be linted
pub fn func2(a: bool) -> Option<i32> {
    if a {
        Some(1)
    } else {
        Some(1)
    }
}

// should not be linted
fn func3(a: bool) -> Option<i32> {
    if a {
        Some(1)
    } else {
        None
    }
}

// should be linted
fn func4() -> Option<i32> {
    Some(1)
}

// should be linted
fn func5() -> Result<i32, ()> {
    Ok(1)
}

// should not be linted
fn func6(a: bool) -> Result<i32, ()> {
    if a {
        Ok(1)
    } else {
        Err(())
    }
}

fn main() {
    // method calls are not linted
    func1(true, true);
    func2(true);
}
