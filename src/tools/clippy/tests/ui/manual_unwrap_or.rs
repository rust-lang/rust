// run-rustfix
#![allow(dead_code)]

fn unwrap_or() {
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

fn main() {}
