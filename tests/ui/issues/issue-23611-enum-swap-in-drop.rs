// run-pass
#![allow(non_upper_case_globals)]

// Issue 23611: this test is ensuring that, for an instance `X` of the
// enum `E`, if you swap in a different variant during the execution
// of the `<E as Drop>::drop`, then the appropriate substructure will
// be torn down after the `<E as Drop>::drop` method returns.

use std::cell::RefCell;
use std::mem;

use self::d::D;

pub fn main() {
    let log = RefCell::new(vec![]);
    d::println("created empty log");
    test(&log);

    // println!("log: {:?}", &log.borrow()[..]);
    assert_eq!(
        &log.borrow()[..],
        [
            //                                         created empty log
            // +-- Make D(test_1, 10000000)
            // | +-- Make D(g_b_5, 50000001)
            // | |                                     in g_B(b2b0) from E::drop, b=b4b0
            // | +-- Drop D(g_b_5, 50000001)
            50000001,
            // |
            // | +-- Make D(drop_6, 60000002)
            // | | +-- Make D(g_b_5, 50000003)
            // | | |                                   in g_B(b2b0) from E::drop, b=b4b1
            // | | +-- Drop D(g_b_5, 50000003)
            50000003,
            // | |
            // | | +-- Make D(GaspB::drop_3, 30000004)
            // | | | +-- Make D(g_b_5, 50000005)
            // | | | |                                 in g_B(b4b2) from GaspB::drop
            // | | | +-- Drop D(g_b_5, 50000005)
            50000005,
            // | | |
            // | | +-- Drop D(GaspB::drop_3, 30000004)
            30000004,
            // | |
            // +-- Drop D(test_1, 10000000)
            10000000,
            //   |
            // +-- Make D(GaspA::drop_2, 20000006)
            // | | +-- Make D(f_a_4, 40000007)
            // | | |                                   in f_A(a3a0) from GaspA::drop
            // | | +-- Drop D(f_a_4, 40000007)
            40000007,
            // | |
            // +-- Drop D(GaspA::drop_2, 20000006)
            20000006,
            //   |
            //   +-- Drop D(drop_6, 60000002)
            60000002
            //
                ]);

    // For reference purposes, the old (incorrect) behavior would produce the following
    // output, which you can compare to the above:
    //
    //                                             created empty log
    // +-- Make D(test_1, 10000000)
    // | +-- Make D(g_b_5, 50000001)
    // | |                                     in g_B(b2b0) from E::drop, b=b4b0
    // | +-- Drop D(g_b_5, 50000001)
    // |
    // | +-- Make D(drop_6, 60000002)
    // | | +-- Make D(g_b_5, 50000003)
    // | | |                                   in g_B(b2b0) from E::drop, b=b4b1
    // | | +-- Drop D(g_b_5, 50000003)
    // | |
    // | | +-- Make D(GaspB::drop_3, 30000004)
    // | | | +-- Make D(g_b_5, 50000005)
    // | | | |                                 in g_B(b4b2) from GaspB::drop
    // | | | +-- Drop D(g_b_5, 50000005)
    // | | |
    // | | +-- Drop D(GaspB::drop_3, 30000004)
    // | |
    // +-- Drop D(test_1, 10000000)
    //   |
    // +-- Make D(GaspB::drop_3, 30000006)
    // | | +-- Make D(f_a_4, 40000007)
    // | | |                                   in f_A(a3a0) from GaspB::drop
    // | | +-- Drop D(f_a_4, 40000007)
    // | |
    // +-- Drop D(GaspB::drop_3, 30000006)
    //   |
    //   +-- Drop D(drop_6, 60000002)

    // Note that this calls f_A from GaspB::drop (and thus creates a D
    // with a uid incorporating the origin of GaspB's drop that
    // surrounds the f_A invocation), but the code as written only
    // ever hands f_A off to instances of GaspA, and thus one should
    // be able to prove the invariant that f_A is *only* invoked from
    // from an instance of GaspA (either via the GaspA drop
    // implementation or the E drop implementation). Yet the old (bad)
    // behavior allowed a call to f_A to leak in while we are tearing
    // down a value of type GaspB.
}

fn test<'a>(log: d::Log<'a>) {
    let _e = E::B(GaspB(g_b, 0xB4B0, log, D::new("test", 1, log)), true);
}

struct GaspA<'a>(for <'b> fn(u32, &'b str, d::Log<'a>), u32, d::Log<'a>, d::D<'a>);
struct GaspB<'a>(for <'b> fn(u32, &'b str, d::Log<'a>), u32, d::Log<'a>, d::D<'a>);

impl<'a> Drop for GaspA<'a> {
    fn drop(&mut self) {
        let _d = d::D::new("GaspA::drop", 2, self.2);
        (self.0)(self.1, "GaspA::drop", self.2);
    }
}

impl<'a> Drop for GaspB<'a> {
    fn drop(&mut self) {
        let _d = d::D::new("GaspB::drop", 3, self.2);
        (self.0)(self.1, "GaspB::drop", self.2);
    }
}

enum E<'a> {
    A(GaspA<'a>, bool), B(GaspB<'a>, bool),
}

fn f_a(x: u32, ctxt: &str, log: d::Log) {
    let _d = d::D::new("f_a", 4, log);
    d::println(&format!("in f_A({:x}) from {}", x, ctxt));
}
fn g_b(y: u32, ctxt: &str, log: d::Log) {
    let _d = d::D::new("g_b", 5, log);
    d::println(&format!("in g_B({:x}) from {}", y, ctxt));
}

impl<'a> Drop for E<'a> {
    fn drop(&mut self) {
        let (do_drop, log) = match *self {
            E::A(GaspA(ref f, ref mut val_a, log, ref _d_a), ref mut do_drop) => {
                f(0xA1A0, &format!("E::drop, a={:x}", val_a), log);
                *val_a += 1;
                // swap in do_drop := false to avoid infinite dtor regress
                (mem::replace(do_drop, false), log)
            }
            E::B(GaspB(ref g, ref mut val_b, log, ref _d_b), ref mut do_drop) => {
                g(0xB2B0, &format!("E::drop, b={:x}", val_b), log);
                *val_b += 1;
                // swap in do_drop := false to avoid infinite dtor regress
                (mem::replace(do_drop, false), log)
            }
        };

        #[allow(unused_must_use)]
        if do_drop {
            mem::replace(self, E::A(GaspA(f_a, 0xA3A0, log, D::new("drop", 6, log)), true));
        }
    }
}

// This module provides simultaneous printouts of the dynamic extents
// of all of the D values, in addition to logging the order that each
// is dropped.
//
// This code is similar to a support code module embedded within
// test/run-pass/issue-123338-ensure-param-drop-order.rs; however,
// that (slightly simpler) code only identifies objects in the log via
// (creation) time-stamps; this incorporates both timestamping and the
// point of origin within the source code into the unique ID (uid).

const PREF_INDENT: u32 = 20;

pub mod d {
    #![allow(unused_parens)]
    use std::fmt;
    use std::mem;
    use std::cell::RefCell;

    static mut counter: u16 = 0;
    static mut trails: u64 = 0;

    pub type Log<'a> = &'a RefCell<Vec<u32>>;

    pub fn current_width() -> u32 {
        unsafe { max_width() - trails.leading_zeros() }
    }

    pub fn max_width() -> u32 {
        unsafe {
            (mem::size_of_val(&trails)*8) as u32
        }
    }

    pub fn indent_println(my_trails: u32, s: &str) {
        let mut indent: String = String::new();
        for i in 0..my_trails {
            unsafe {
                if trails & (1 << i) != 0 {
                    indent = indent + "| ";
                } else {
                    indent = indent + "  ";
                }
            }
        }
        println!("{}{}", indent, s);
    }

    pub fn println(s: &str) {
        indent_println(super::PREF_INDENT, s);
    }

    fn first_avail() -> u32 {
        unsafe {
            for i in 0..64 {
                if trails & (1 << i) == 0 {
                    return i;
                }
            }
        }
        panic!("exhausted trails");
    }

    pub struct D<'a> {
        name: &'static str, i: u8, uid: u32, trail: u32, log: Log<'a>
    }

    impl<'a> fmt::Display for D<'a> {
        fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result {
            write!(w, "D({}_{}, {})", self.name, self.i, self.uid)
        }
    }

    impl<'a> D<'a> {
        pub fn new(name: &'static str, i: u8, log: Log<'a>) -> D<'a> {
            unsafe {
                let trail = first_avail();
                let ctr = ((i as u32) * 10_000_000) + (counter as u32);
                counter += 1;
                trails |= (1 << trail);
                let ret = D {
                    name: name, i: i, log: log, uid: ctr, trail: trail
                };
                indent_println(trail, &format!("+-- Make {}", ret));
                ret
            }
        }
    }

    impl<'a> Drop for D<'a> {
        fn drop(&mut self) {
            unsafe { trails &= !(1 << self.trail); };
            self.log.borrow_mut().push(self.uid);
            indent_println(self.trail, &format!("+-- Drop {}", self));
            indent_println(::PREF_INDENT, "");
        }
    }
}
