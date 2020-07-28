// needs-sanitizer-support
// needs-sanitizer-leak
//
// compile-flags: -Z sanitizer=leak -O
//
// run-fail
// error-pattern: LeakSanitizer: detected memory leaks

#![feature(test)]

use std::hint::pretend_used;
use std::mem;

fn main() {
    for _ in 0..10 {
        let xs = vec![1, 2, 3];
        // Prevent compiler from removing the memory allocation.
        let xs = pretend_used(xs);
        mem::forget(xs);
    }
}
