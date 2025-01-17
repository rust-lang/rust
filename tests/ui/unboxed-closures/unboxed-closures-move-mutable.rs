//@ run-pass

#![deny(unused_mut)]
#![allow(unused_must_use)]

// Test that mutating a mutable upvar in a capture-by-value unboxed
// closure does not ice (issue #18238) and marks the upvar as used
// mutably so we do not get a spurious warning about it not needing to
// be declared mutable (issue #18336 and #18769)

fn set(x: &mut usize) { *x = 42; }

fn main() {
    {
        let mut x = 0_usize;
        move || x += 1; //~ WARN unused variable: `x`
    }
    {
        let mut x = 0_usize;
        move || x += 1; //~ WARN unused variable: `x`
    }
    {
        let mut x = 0_usize;
        move || set(&mut x);
    }
    {
        let mut x = 0_usize;
        move || set(&mut x);
    }
}
