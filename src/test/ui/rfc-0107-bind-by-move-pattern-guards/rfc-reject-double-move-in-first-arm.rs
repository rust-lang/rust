#![feature(nll)]
#![feature(bind_by_move_pattern_guards)]

struct A { a: Box<i32> }

fn foo(n: i32) {
    let x = A { a: Box::new(n) };
    let _y = match x {
        A { a: v } if { drop(v); true } => v,
        //~^ ERROR cannot move out of `v` in pattern guard
        _ => Box::new(0),
    };
}

fn main() {
    foo(107);
}
