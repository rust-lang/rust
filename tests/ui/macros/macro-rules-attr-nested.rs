//@ run-pass
//@ check-run-results
#![feature(macro_attr)]

macro_rules! nest {
    attr() { struct $name:ident; } => {
        println!("nest");
        #[nest(1)]
        struct $name;
    };
    attr(1) { struct $name:ident; } => {
        println!("nest(1)");
        #[nest(2)]
        struct $name;
    };
    attr(2) { struct $name:ident; } => {
        println!("nest(2)");
    };
}

fn main() {
    #[nest]
    struct S;
}
