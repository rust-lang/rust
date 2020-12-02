#![allow(unused_assignments, unused_variables)]

use std::fmt::Debug;

pub fn used_function() {
    // Initialize test constants in a way that cannot be determined at compile time, to ensure
    // rustc and LLVM cannot optimize out statements (or coverage counters) downstream from
    // dependent conditions.
    let is_true = std::env::args().len() == 1;
    let mut countdown = 0;
    if is_true {
        countdown = 10;
    }
    used_twice_generic_function("some str");
}

pub fn used_generic_function<T: Debug>(arg: T) {
    println!("used_generic_function with {:?}", arg);
}

pub fn used_twice_generic_function<T: Debug>(arg: T) {
    println!("used_twice_generic_function with {:?}", arg);
}

pub fn unused_generic_function<T: Debug>(arg: T) {
    println!("unused_generic_function with {:?}", arg);
}

pub fn unused_function() {
    let is_true = std::env::args().len() == 1;
    let mut countdown = 2;
    if !is_true {
        countdown = 20;
    }
}

fn unused_private_function() {
    let is_true = std::env::args().len() == 1;
    let mut countdown = 2;
    if !is_true {
        countdown = 20;
    }
}
