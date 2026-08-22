//! `Type { .. }` patterns get translated into wildcards,
//! so restrictions on mixing deref and non-deref patterns
//! don't apply to them.
//! (And you can't name fields of `Box` outside `alloc`.)

//@ run-pass

#![feature(box_patterns)]

fn main() {
    match Box::new(0) {
        box _ => {}
        Box { .. } => {}
        //~^ WARN unreachable pattern
    }
}
