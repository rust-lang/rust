#![warn(clippy::if_then_some_else_none)]
#![allow(clippy::redundant_pattern_matching, clippy::unnecessary_lazy_evaluations)]

fn main() {
    // Should issue an error.
    let _ = if foo() {
        //~^ ERROR: this could be simplified with `bool::then`
        println!("true!");
        Some("foo")
    } else {
        None
    };

    // Should issue an error when macros are used.
    let _ = if matches!(true, true) {
        //~^ ERROR: this could be simplified with `bool::then`
        println!("true!");
        Some(matches!(true, false))
    } else {
        None
    };

    // Should issue an error. Binary expression `o < 32` should be parenthesized.
    let x = Some(5);
    let _ = x.and_then(|o| if o < 32 { Some(o) } else { None });
    //~^ ERROR: this could be simplified with `bool::then_some`

    // Should issue an error. Unary expression `!x` should be parenthesized.
    let x = true;
    let _ = if !x { Some(0) } else { None };
    //~^ ERROR: this could be simplified with `bool::then_some`

    // Should not issue an error since the `else` block has a statement besides `None`.
    let _ = if foo() {
        println!("true!");
        Some("foo")
    } else {
        eprintln!("false...");
        None
    };

    // Should not issue an error since there are more than 2 blocks in the if-else chain.
    let _ = if foo() {
        println!("foo true!");
        Some("foo")
    } else if bar() {
        println!("bar true!");
        Some("bar")
    } else {
        None
    };

    let _ = if foo() {
        println!("foo true!");
        Some("foo")
    } else {
        bar().then(|| {
            println!("bar true!");
            "bar"
        })
    };

    // Should not issue an error since the `then` block has `None`, not `Some`.
    let _ = if foo() { None } else { Some("foo is false") };

    // Should not issue an error since the `else` block doesn't use `None` directly.
    let _ = if foo() { Some("foo is true") } else { into_none() };

    // Should not issue an error since the `then` block doesn't use `Some` directly.
    let _ = if foo() { into_some("foo") } else { None };
}

#[clippy::msrv = "1.49"]
fn _msrv_1_49() {
    // `bool::then` was stabilized in 1.50. Do not lint this
    let _ = if foo() {
        println!("true!");
        Some(149)
    } else {
        None
    };
}

#[clippy::msrv = "1.50"]
fn _msrv_1_50() {
    let _ = if foo() {
        //~^ ERROR: this could be simplified with `bool::then`
        println!("true!");
        Some(150)
    } else {
        None
    };
}

fn foo() -> bool {
    unimplemented!()
}

fn bar() -> bool {
    unimplemented!()
}

fn into_some<T>(v: T) -> Option<T> {
    Some(v)
}

fn into_none<T>() -> Option<T> {
    None
}

// Should not warn
fn f(b: bool, v: Option<()>) -> Option<()> {
    if b {
        v?; // This is a potential early return, is not equivalent with `bool::then`

        Some(())
    } else {
        None
    }
}

fn issue11394(b: bool, v: Result<(), ()>) -> Result<(), ()> {
    let x = if b {
        #[allow(clippy::let_unit_value)]
        let _ = v?;
        Some(())
    } else {
        None
    };

    Ok(())
}

fn issue13407(s: &str) -> Option<bool> {
    if s == "1" { Some(true) } else { None }
}

const fn issue12103(x: u32) -> Option<u32> {
    // Should not issue an error in `const` context
    if x > 42 { Some(150) } else { None }
}
