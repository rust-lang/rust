#![warn(clippy::semicolon_if_nothing_returned)]
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
