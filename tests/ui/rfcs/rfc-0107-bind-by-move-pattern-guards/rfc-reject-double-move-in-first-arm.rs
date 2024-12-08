#![feature(if_let_guard)]

struct A { a: Box<i32> }

fn if_guard(n: i32) {
    let x = A { a: Box::new(n) };
    let _y = match x {
        A { a: v } if { drop(v); true } => v,
        //~^ ERROR cannot move out of `v` in pattern guard
        _ => Box::new(0),
    };
}

fn if_let_guard(n: i32) {
    let x = A { a: Box::new(n) };
    let _y = match x {
        A { a: v } if let Some(()) = { drop(v); Some(()) } => v,
        //~^ ERROR cannot move out of `v` in pattern guard
        _ => Box::new(0),
    };
}

fn main() {
    if_guard(107);
    if_let_guard(107);
}
