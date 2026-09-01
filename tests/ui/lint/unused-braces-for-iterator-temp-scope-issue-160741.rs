//! A block around a Rust 2024 `for` iterator expression may shorten the lifetime of temporaries.

//@ edition: 2024
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
        //~^ WARN unnecessary braces around `for` iterator expression
        data.lock().unwrap().total += value;
    }
}
