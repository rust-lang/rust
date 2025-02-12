#![warn(clippy::single_option_map)]

use std::sync::atomic::{AtomicUsize, Ordering};

static ATOM: AtomicUsize = AtomicUsize::new(42);
static MAYBE_ATOMIC: Option<&AtomicUsize> = Some(&ATOM);

fn h(arg: Option<u32>) -> Option<u32> {
    //~^ single_option_map

    arg.map(|x| x * 2)
}

fn j(arg: Option<u64>) -> Option<u64> {
    //~^ single_option_map

    arg.map(|x| x * 2)
}

fn mul_args(a: String, b: u64) -> String {
    a
}

fn mul_args_opt(a: Option<String>, b: u64) -> Option<String> {
    //~^ single_option_map

    a.map(|val| mul_args(val, b + 1))
}

// No lint: no `Option` argument argument
fn maps_static_option() -> Option<usize> {
    MAYBE_ATOMIC.map(|a| a.load(Ordering::Relaxed))
}

// No lint: wrapped by another function
fn manipulate(i: i32) -> i32 {
    i + 1
}
// No lint: wraps another function to do the optional thing
fn manipulate_opt(opt_i: Option<i32>) -> Option<i32> {
    opt_i.map(manipulate)
}

// No lint: maps other than the receiver
fn map_not_arg(arg: Option<u32>) -> Option<u32> {
    maps_static_option().map(|_| arg.unwrap())
}

// No lint: wrapper function with η-expanded form
#[allow(clippy::redundant_closure)]
fn manipulate_opt_explicit(opt_i: Option<i32>) -> Option<i32> {
    opt_i.map(|x| manipulate(x))
}

// No lint
fn multi_args(a: String, b: bool, c: u64) -> String {
    a
}

// No lint: contains only map of a closure that binds other arguments
fn multi_args_opt(a: Option<String>, b: bool, c: u64) -> Option<String> {
    a.map(|a| multi_args(a, b, c))
}

fn main() {
    let answer = Some(42u32);
    let h_result = h(answer);

    let answer = Some(42u64);
    let j_result = j(answer);
    maps_static_option();
}
