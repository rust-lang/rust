//@aux-build:proc_macros.rs
#![warn(clippy::reserve_after_initialization)]
#![no_main]

extern crate proc_macros;
use proc_macros::{external, with_span};

// Should lint
fn standard() {
    let mut v1: Vec<usize> = vec![];
    //~^ reserve_after_initialization
    v1.reserve(10);
}

// Should lint
fn capacity_as_expr() {
    let capacity = 10;
    let mut v2: Vec<usize> = vec![];
    //~^ reserve_after_initialization
    v2.reserve(capacity);
}

// Shouldn't lint
fn vec_init_with_argument() {
    let mut v3 = vec![1];
    v3.reserve(10);
}

// Shouldn't lint
fn called_with_capacity() {
    let _v4: Vec<usize> = Vec::with_capacity(10);
}

// Should lint
fn assign_expression() {
    let mut v5: Vec<usize> = Vec::new();
    v5 = Vec::new();
    //~^ reserve_after_initialization
    v5.reserve(10);
}

fn in_macros() {
    external! {
        let mut v: Vec<usize> = vec![];
        v.reserve(10);
    }

    with_span! {
        span

        let mut v: Vec<usize> = vec![];
        v.reserve(10);
    }
}
