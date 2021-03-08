#![warn(clippy::if_then_some_else_none)]
#![feature(custom_inner_attributes)]

fn main() {
    // Should issue an error.
    let _ = if foo() {
        println!("true!");
        Some("foo")
    } else {
        None
    };

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

fn _msrv_1_49() {
    #![clippy::msrv = "1.49"]
    // `bool::then` was stabilized in 1.50. Do not lint this
    let _ = if foo() {
        println!("true!");
        Some("foo")
    } else {
        None
    };
}

fn _msrv_1_50() {
    #![clippy::msrv = "1.50"]
    let _ = if foo() {
        println!("true!");
        Some("foo")
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
