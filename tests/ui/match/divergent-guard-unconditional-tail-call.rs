//! Companion test to `divergent-guard-unconditional.rs`, separated out for the use of
//! `#![feature(explicit_tail_calls)]`: it's an error to mutate a matched-on place in a guard, even
//! if the guard unconditionally diverges by explicit tail call thereafter.

#![feature(explicit_tail_calls)]

fn main() {
    let mut x: Option<&u64> = Some(&7);
    match x {
        Some(&y) if { x = None; become main() } => {}
        //~^ ERROR: cannot assign `x` in match guard

        Some(&y) => {}
        None => {},
    }
}
