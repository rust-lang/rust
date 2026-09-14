//@ check-pass
//@ run-rustfix

// Removing block braces can change how macro-generated calls match macro arguments.

#![warn(unused_braces)]

macro_rules! type_or_expr {
    ($x:ty) => {
        identity(stringify!($x))
    };
    ($x:expr) => {
        identity($x)
    };
}

macro_rules! call_expr {
    ($e:expr) => {
        identity($e)
    };
}

macro_rules! call_block {
    ($b:block) => {
        identity($b)
    };
}

fn identity<T>(x: T) -> T {
    x
}

fn main() {
    // should not warn
    let _ = type_or_expr!({ format!("{}", 1) });
    // should not warn
    let _ = call_expr!({ 1 });
    //~^ WARN unnecessary braces around function argument
    let _ = call_block!({ 1 });
}
