#![warn(clippy::semicolon_if_nothing_returned)]
#![allow(clippy::redundant_closure)]
#![feature(label_break_value)]

fn get_unit() {}

// the functions below trigger the lint
fn main() {
    println!("Hello")
}

fn hello() {
    get_unit()
}

fn basic101(x: i32) {
    let y: i32;
    y = x + 1
}

#[rustfmt::skip]
fn closure_error() {
    let _d = || {
        hello()
    };
}

#[rustfmt::skip]
fn unsafe_checks_error() {
    use std::mem::MaybeUninit;
    use std::ptr;

    let mut s = MaybeUninit::<String>::uninit();
    let _d = || unsafe {
        ptr::drop_in_place(s.as_mut_ptr())
    };
}

// this is fine
fn print_sum(a: i32, b: i32) {
    println!("{}", a + b);
    assert_eq!(true, false);
}

fn foo(x: i32) {
    let y: i32;
    if x < 1 {
        y = 4;
    } else {
        y = 5;
    }
}

fn bar(x: i32) {
    let y: i32;
    match x {
        1 => y = 4,
        _ => y = 32,
    }
}

fn foobar(x: i32) {
    let y: i32;
    'label: {
        y = x + 1;
    }
}

fn loop_test(x: i32) {
    let y: i32;
    for &ext in &["stdout", "stderr", "fixed"] {
        println!("{}", ext);
    }
}

fn closure() {
    let _d = || hello();
}

#[rustfmt::skip]
fn closure_block() {
    let _d = || { hello() };
}

unsafe fn some_unsafe_op() {}
unsafe fn some_other_unsafe_fn() {}

fn do_something() {
    unsafe { some_unsafe_op() };

    unsafe { some_other_unsafe_fn() };
}

fn unsafe_checks() {
    use std::mem::MaybeUninit;
    use std::ptr;

    let mut s = MaybeUninit::<String>::uninit();
    let _d = || unsafe { ptr::drop_in_place(s.as_mut_ptr()) };
}

// Issue #7768
#[rustfmt::skip]
fn macro_with_semicolon() {
    macro_rules! repro {
        () => {
            while false {
            }
        };
    }
    repro!();
}
