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
    match Some(1) {
        Some(i) => i,
        None => {
            let a = 1 + 42;
            let b = a + 42;
            b + 42
        },
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
}

fn main() {}
