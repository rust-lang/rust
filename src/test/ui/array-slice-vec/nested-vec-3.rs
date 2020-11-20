// run-pass
#![allow(overflowing_literals)]

// ignore-emscripten no threads support

// Test that using the `vec!` macro nested within itself works when
// the contents implement Drop and we hit a panic in the middle of
// construction.

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
            Ordering::SeqCst,
        );
    }
}

fn main() {
    fn die() -> D { panic!("Oh no"); }
    let g = thread::spawn(|| {
        let _nested = vec![vec![D( 1), D( 2), D( 3), D( 4)],
                           vec![D( 5), D( 6), D( 7), D( 8)],
                           vec![D( 9), D(10), die(), D(12)],
                           vec![D(13), D(14), D(15), D(16)]];
    });
    assert!(g.join().is_err());

    // When the panic occurs, we will be in the midst of constructing the
    // second inner vector.  Therefore, we drop the elements of the
    // partially filled vector first, before we get around to dropping
    // the elements of the filled vector.

    // Issue 23222: The order in which the elements actually get
    // dropped is a little funky: as noted above, we'll drop the 9+10
    // first, but due to #23222, they get dropped in reverse
    // order. Likewise, again due to #23222, we will drop the second
    // filled vec before the first filled vec.
    //
    // If Issue 23222 is "fixed", then presumably the corrected
    // expected order of events will be 0x__9_A__1_2_3_4__5_6_7_8;
    // that is, we would still drop 9+10 first, since they belong to
    // the more deeply nested expression when the panic occurs.

    let expect = 0x__A_9__5_6_7_8__1_2_3_4;
    let actual = LOG.load(Ordering::SeqCst);
    assert!(actual == expect, "expect: 0x{:x} actual: 0x{:x}", expect, actual);
}
