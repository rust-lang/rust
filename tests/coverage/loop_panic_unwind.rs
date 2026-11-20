#![feature(coverage_attribute)]
//@ edition: 2024
//@ needs-unwind

// Regression test for <https://github.com/rust-lang/rust/issues/164153>.
//
// It might so happen that a loop doesn't exit before the coverage is dumped.
// So, we cannot assume every loop execution eventually takes a normal loop
// exit. Otherwise we will end up with negative counter values based on the
// expression evaluation.

use std::hint::black_box;

fn branch_then_two_match_loop() {
    if black_box(false) {
        return;
    }
    let mut count = 0;
    loop {
        let should_return = match step(&mut count) {
            Ok(()) => continue,
            Err(should_return) => should_return,
        };
        match should_return {
            false => continue,
            true => return,
        }
    }
}

fn inescapable_loop() -> ! {
    let mut count = 0;
    loop {
        let _ = step(&mut count);
        if black_box(true) {
            black_box(1);
        } else {
            black_box(2);
        }
    }
}

#[coverage(off)]
#[inline(never)]
fn step(count: &mut usize) -> Result<(), bool> {
    *count += 1;
    if *count == 3 {
        panic!("stop loop");
    }
    Ok(())
}

#[coverage(off)]
fn main() {
    assert!(std::panic::catch_unwind(branch_then_two_match_loop).is_err());
    assert!(std::panic::catch_unwind(inescapable_loop).is_err());
}
