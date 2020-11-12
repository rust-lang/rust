// run-rustfix
#![allow(dead_code)]
#![allow(unused_variables)]

fn option_unwrap_or() {
    // int case
    match Some(1) {
        Some(i) => i,
        None => 42,
    };

    // int case reversed
    match Some(1) {
        None => 42,
        Some(i) => i,
    };

    // richer none expr
    match Some(1) {
        Some(i) => i,
        None => 1 + 42,
    };

    // multiline case
    #[rustfmt::skip]
    match Some(1) {
        Some(i) => i,
        None => {
            42 + 42
                + 42 + 42 + 42
                + 42 + 42 + 42
        }
    };

    // string case
    match Some("Bob") {
        Some(i) => i,
        None => "Alice",
    };

    // don't lint
    match Some(1) {
        Some(i) => i + 2,
        None => 42,
    };
    match Some(1) {
        Some(i) => i,
        None => return,
    };
    for j in 0..4 {
        match Some(j) {
            Some(i) => i,
            None => continue,
        };
        match Some(j) {
            Some(i) => i,
            None => break,
        };
    }

    // cases where the none arm isn't a constant expression
    // are not linted due to potential ownership issues

    // ownership issue example, don't lint
    struct NonCopyable;
    let mut option: Option<NonCopyable> = None;
    match option {
        Some(x) => x,
        None => {
            option = Some(NonCopyable);
            // some more code ...
            option.unwrap()
        },
    };

    // ownership issue example, don't lint
    let option: Option<&str> = None;
    match option {
        Some(s) => s,
        None => &format!("{} {}!", "hello", "world"),
    };
}

fn result_unwrap_or() {
    // int case
    match Ok::<i32, &str>(1) {
        Ok(i) => i,
        Err(_) => 42,
    };

    // int case, scrutinee is a binding
    let a = Ok::<i32, &str>(1);
    match a {
        Ok(i) => i,
        Err(_) => 42,
    };

    // int case, suggestion must surround Result expr with parenthesis
    match Ok(1) as Result<i32, &str> {
        Ok(i) => i,
        Err(_) => 42,
    };

    // method call case, suggestion must not surround Result expr `s.method()` with parenthesis
    struct S {}
    impl S {
        fn method(self) -> Option<i32> {
            Some(42)
        }
    }
    let s = S {};
    match s.method() {
        Some(i) => i,
        None => 42,
    };

    // int case reversed
    match Ok::<i32, &str>(1) {
        Err(_) => 42,
        Ok(i) => i,
    };

    // richer none expr
    match Ok::<i32, &str>(1) {
        Ok(i) => i,
        Err(_) => 1 + 42,
    };

    // multiline case
    #[rustfmt::skip]
    match Ok::<i32, &str>(1) {
        Ok(i) => i,
        Err(_) => {
            42 + 42
                + 42 + 42 + 42
                + 42 + 42 + 42
        }
    };

    // string case
    match Ok::<&str, &str>("Bob") {
        Ok(i) => i,
        Err(_) => "Alice",
    };

    // don't lint
    match Ok::<i32, &str>(1) {
        Ok(i) => i + 2,
        Err(_) => 42,
    };
    match Ok::<i32, &str>(1) {
        Ok(i) => i,
        Err(_) => return,
    };
    for j in 0..4 {
        match Ok::<i32, &str>(j) {
            Ok(i) => i,
            Err(_) => continue,
        };
        match Ok::<i32, &str>(j) {
            Ok(i) => i,
            Err(_) => break,
        };
    }

    // don't lint, Err value is used
    match Ok::<&str, &str>("Alice") {
        Ok(s) => s,
        Err(s) => s,
    };
    // could lint, but unused_variables takes care of it
    match Ok::<&str, &str>("Alice") {
        Ok(s) => s,
        Err(s) => "Bob",
    };
}

fn main() {}
