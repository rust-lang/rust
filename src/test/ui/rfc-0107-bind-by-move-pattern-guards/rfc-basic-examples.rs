#![feature(bind_by_move_pattern_guards)]

// build-pass (FIXME(62277): could be check-pass?)

struct A { a: Box<i32> }

impl A {
    fn get(&self) -> i32 { *self.a }
}

fn foo(n: i32) {
    let x = A { a: Box::new(n) };
    let y = match x {
        A { a: v } if *v == 42 => v,
        _ => Box::new(0),
    };
}

fn bar(n: i32) {
    let x = A { a: Box::new(n) };
    let y = match x {
        A { a: v } if x.get() == 42 => v,
        _ => Box::new(0),
    };
}

fn baz(n: i32) {
    let x = A { a: Box::new(n) };
    let y = match x {
        A { a: v } if *v.clone() == 42 => v,
        _ => Box::new(0),
    };
}

fn main() {
    foo(107);
    bar(107);
    baz(107);
}
