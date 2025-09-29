//! Some temporaries are implemented as local variables bound with `super let`. These can be
//! lifetime-extended, and as such are subject to shortening after #145838.
//@ edition: 2024
//@ check-pass

fn main() {
    // The `()` argument to the inner `format_args!` is promoted, but the lifetimes of the internal
    // `super let` temporaries in its expansion shorten, making this an error in Rust 1.92.
    println!("{:?}{}", (), { format_args!("{:?}", ()) });
    // TODO: warn
}
