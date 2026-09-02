//! A block around a Rust 2024 `for` iterator expression may shorten the lifetime of temporaries.

//@ revisions: e2021 e2024
//@[e2021] edition: 2021
//@[e2024] edition: 2024
//@ check-pass

#![warn(unused_braces)]

use std::sync::{Arc, Mutex};

struct State {
    values: Vec<u32>,
    total: u32,
}

fn main() {
    let data = Arc::new(Mutex::new(State { values: vec![1, 2, 3], total: 0 }));

    for value in { data.lock().unwrap().values.clone() } {
        //[e2021]~^ WARN unnecessary braces around `for` iterator expression
        data.lock().unwrap().total += value;
    }
}
