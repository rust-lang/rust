//! Companion test to `divergent-guard-unconditional.rs`, separated out for the use of
//! `#![feature(explicit_tail_calls)]`: it's fine for a match guard to mutate matched-on places
//! within the scrutinee as long as it unconditionally diverges thereafter, including by explicit
//! tail call.
//@ check-pass

#![feature(explicit_tail_calls)]

fn main() {
    let mut x: Option<&u64> = Some(&7);
    match x {
        Some(&y) if { x = None; become main() } => {}

        Some(&y) => {}
        None => {},
    }
}
