#![allow(dead_code)]
#![deny(unused_assignments)]

fn f1(x: &mut isize) {
    *x = 1; // no error
}

fn f2() {
    let mut x: isize = 3; //~ ERROR: value assigned to `x` is never read
    x = 4;
    x.clone();
}

fn f3() {
    let mut x: isize = 3;
    x.clone();
    x = 4; //~ ERROR: value assigned to `x` is never read
}

fn f4(mut x: i32) { //~ ERROR: value passed to `x` is never read
    x = 4;
    x.clone();
}

fn f5(mut x: i32) {
    x.clone();
    x = 4; //~ ERROR: value assigned to `x` is never read
}

// #22630
fn f6() {
    let mut done = false;
    while !done {
        done = true; // no error
        continue;
    }
}

fn f7(x: i32) {
    match x {
        n if n > 22 => {} // no error
        _ => {}
    }
}

fn f8(x: Option<i32>) -> i32 {
    match x {
        None => 0,
        Some(mut n) => {
            //~^ ERROR: value assigned to `n` is never read
            n = 42;
            n
        }
    }
}

fn main() {}
