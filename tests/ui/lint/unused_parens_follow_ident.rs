//@ run-rustfix

#![deny(unused_parens)]

macro_rules! wrap {
    ($name:ident $arg:expr) => {
        $name($arg);
    };
}

fn main() {
    wrap!(unary(routine())); //~ ERROR unnecessary parentheses around function argument
    wrap!(unary (routine())); //~ ERROR unnecessary parentheses around function argument
}

fn unary(_: ()) {}
fn routine() {}
