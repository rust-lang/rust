// run-pass
// needs-unwind
#![allow(overflowing_literals)]

// Test that we cleanup dynamic sized Box<[D]> properly when D has a
// destructor.

// ignore-emscripten no threads support

use std::thread;
use std::sync::atomic::{AtomicUsize, Ordering};

static LOG: AtomicUsize = AtomicUsize::new(0);

struct D(u8);

impl Drop for D {
    fn drop(&mut self) {
        println!("Dropping {}", self.0);
        let old = LOG.load(Ordering::SeqCst);
        let _ = LOG.compare_exchange(
            old,
            old << 4 | self.0 as usize,
            Ordering::SeqCst,
            Ordering::SeqCst
        );
    }
}

fn main() {
    fn die() -> D { panic!("Oh no"); }
    let g = thread::spawn(|| {
        let _b1: Box<[D; 4]> = Box::new([D( 1), D( 2), D( 3), D( 4)]);
        let _b2: Box<[D; 4]> = Box::new([D( 5), D( 6), D( 7), D( 8)]);
        let _b3: Box<[D; 4]> = Box::new([D( 9), D(10), die(), D(12)]);
        let _b4: Box<[D; 4]> = Box::new([D(13), D(14), D(15), D(16)]);
    });
    assert!(g.join().is_err());

    // When the panic occurs, we will be in the midst of constructing
    // the input to `_b3`.  Therefore, we drop the elements of the
    // partially filled array first, before we get around to dropping
    // the elements of `_b1` and _b2`.

    // Issue 23222: The order in which the elements actually get
    // dropped is a little funky. See similar notes in nested-vec-3;
    // in essence, I would not be surprised if we change the ordering
    // given in `expect` in the future.

    let expect = 0x__A_9__5_6_7_8__1_2_3_4;
    let actual = LOG.load(Ordering::SeqCst);
    assert!(actual == expect, "expect: 0x{:x} actual: 0x{:x}", expect, actual);
}
