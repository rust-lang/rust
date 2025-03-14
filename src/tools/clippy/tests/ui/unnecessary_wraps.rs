//@no-rustfix: overlapping suggestions
#![warn(clippy::unnecessary_wraps)]
#![allow(clippy::no_effect)]
#![allow(clippy::needless_return)]
#![allow(clippy::if_same_then_else)]
#![allow(dead_code)]

// should be linted
fn func1(a: bool, b: bool) -> Option<i32> {
    //~^ unnecessary_wraps

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
    //~^ unnecessary_wraps

    if a && b {
        return Some(10);
    }
    if a { Some(20) } else { Some(30) }
}

// public fns should not be linted
pub fn func3(a: bool) -> Option<i32> {
    if a { Some(1) } else { Some(1) }
}

// should not be linted
fn func4(a: bool) -> Option<i32> {
    if a { Some(1) } else { None }
}

// should be linted
fn func5() -> Option<i32> {
    //~^ unnecessary_wraps

    Some(1)
}

// should not be linted
fn func6() -> Option<i32> {
    None
}

// should be linted
fn func7() -> Result<i32, ()> {
    //~^ unnecessary_wraps

    Ok(1)
}

// should not be linted
fn func8(a: bool) -> Result<i32, ()> {
    if a { Ok(1) } else { Err(()) }
}

// should not be linted
fn func9(a: bool) -> Result<i32, ()> {
    Err(())
}

// should not be linted
fn func10() -> Option<()> {
    unimplemented!()
}

pub struct A;

impl A {
    // should not be linted
    pub fn func11() -> Option<i32> {
        Some(1)
    }

    // should be linted
    fn func12() -> Option<i32> {
        //~^ unnecessary_wraps

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

fn issue_6384(s: &str) -> Option<&str> {
    Some(match s {
        "a" => "A",
        _ => return None,
    })
}

// should be linted
fn issue_6640_1(a: bool, b: bool) -> Option<()> {
    //~^ unnecessary_wraps

    if a && b {
        return Some(());
    }
    if a {
        Some(());
        Some(())
    } else {
        return Some(());
    }
}

// should be linted
fn issue_6640_2(a: bool, b: bool) -> Result<(), i32> {
    //~^ unnecessary_wraps

    if a && b {
        return Ok(());
    }
    if a {
        Ok(())
    } else {
        return Ok(());
    }
}

// should not be linted
fn issue_6640_3() -> Option<()> {
    if true { Some(()) } else { None }
}

// should not be linted
fn issue_6640_4() -> Result<(), ()> {
    if true { Ok(()) } else { Err(()) }
}

fn main() {
    // method calls are not linted
    func1(true, true);
    func2(true, true);
    issue_6640_1(true, true);
    issue_6640_2(true, true);
}
