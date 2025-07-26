// Test that dropping values works in match arms, which is nontrivial
// because each match arm needs its own scope.

//@ run-pass

#![allow(incomplete_features)]
#![feature(loop_match)]

use std::sync::atomic::{AtomicBool, Ordering};

fn main() {
    assert_eq!(helper(), 1);
    assert!(DROPPED.load(Ordering::Relaxed));
}

static DROPPED: AtomicBool = AtomicBool::new(false);

struct X;

impl Drop for X {
    fn drop(&mut self) {
        DROPPED.store(true, Ordering::Relaxed);
    }
}

#[no_mangle]
#[inline(never)]
fn helper() -> i32 {
    let mut state = 0;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                0 => match X {
                    _ => {
                        assert!(!DROPPED.load(Ordering::Relaxed));
                        break 'blk 1;
                    }
                },
                _ => {
                    assert!(DROPPED.load(Ordering::Relaxed));
                    break 'a state;
                }
            }
        };
    }
}
