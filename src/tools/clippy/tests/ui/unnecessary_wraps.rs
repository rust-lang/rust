#![warn(clippy::unnecessary_wraps)]
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

// should be linted
fn func2(a: bool, b: bool) -> Option<i32> {
    if a && b {
        return Some(10);
    }
    if a {
        Some(20)
    } else {
        Some(30)
    }
}

// public fns should not be linted
pub fn func3(a: bool) -> Option<i32> {
    if a {
        Some(1)
    } else {
        Some(1)
    }
}

// should not be linted
fn func4(a: bool) -> Option<i32> {
    if a {
        Some(1)
    } else {
        None
    }
}

// should be linted
fn func5() -> Option<i32> {
    Some(1)
}

// should not be linted
fn func6() -> Option<i32> {
    None
}

// should be linted
fn func7() -> Result<i32, ()> {
    Ok(1)
}

// should not be linted
fn func8(a: bool) -> Result<i32, ()> {
    if a {
        Ok(1)
    } else {
        Err(())
    }
}

// should not be linted
fn func9(a: bool) -> Result<i32, ()> {
    Err(())
}

// should not be linted
fn func10() -> Option<()> {
    unimplemented!()
}

struct A;

impl A {
    // should not be linted
    pub fn func11() -> Option<i32> {
        Some(1)
    }

    // should be linted
    fn func12() -> Option<i32> {
        Some(1)
    }
}

trait B {
    // trait impls are not linted
    fn func13() -> Option<i32> {
        Some(1)
    }
}

impl B for A {
    // trait impls are not linted
    fn func13() -> Option<i32> {
        Some(0)
    }
}

fn main() {
    // method calls are not linted
    func1(true, true);
    func2(true, true);
}
