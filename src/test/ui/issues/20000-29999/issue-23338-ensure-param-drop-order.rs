// run-pass
#![allow(non_upper_case_globals)]

// ignore-pretty issue #37201

// This test is ensuring that parameters are indeed dropped after
// temporaries in a fn body.

use std::cell::RefCell;

use self::d::D;

pub fn main() {
    let log = RefCell::new(vec![]);
    d::println("created empty log");
    test(&log);

    assert_eq!(&log.borrow()[..],
               [
                   //                                    created empty log
                   //    +-- Make D(da_0, 0)
                   //    | +-- Make D(de_1, 1)
                   //    | |                             calling foo
                   //    | |                             entered foo
                   //    | | +-- Make D(de_2, 2)
                   //    | | | +-- Make D(da_1, 3)
                   //    | | | | +-- Make D(de_3, 4)
                   //    | | | | | +-- Make D(de_4, 5)
                   3, // | | | +-- Drop D(da_1, 3)
                   //    | | |   | |
                   4, // | | |   +-- Drop D(de_3, 4)
                   //    | | |     |
                   //    | | |     |                     eval tail of foo
                   //    | | | +-- Make D(de_5, 6)
                   //    | | | | +-- Make D(de_6, 7)
                   5, // | | | | | +-- Drop D(de_4, 5)
                   //    | | | | |
                   2, // | | +-- Drop D(de_2, 2)
                   //    | |   | |
                   6, // | |   +-- Drop D(de_5, 6)
                   //    | |     |
                   1, // | +-- Drop D(de_1, 1)
                   //    |       |
                   0, // +-- Drop D(da_0, 0)
                   //            |
                   //            |                       result D(de_6, 7)
                   7 //          +-- Drop D(de_6, 7)

                       ]);
}

fn test<'a>(log: d::Log<'a>) {
    let da = D::new("da", 0, log);
    let de = D::new("de", 1, log);
    d::println("calling foo");
    let result = foo(da, de);
    d::println(&format!("result {}", result));
}

fn foo<'a>(da0: D<'a>, de1: D<'a>) -> D<'a> {
    d::println("entered foo");
    let de2 = de1.incr();      // creates D(de_2, 2)
    let de4 = {
        let _da1 = da0.incr(); // creates D(da_1, 3)
        de2.incr().incr()      // creates D(de_3, 4) and D(de_4, 5)
    };
    d::println("eval tail of foo");
    de4.incr().incr()          // creates D(de_5, 6) and D(de_6, 7)
}

// This module provides simultaneous printouts of the dynamic extents
// of all of the D values, in addition to logging the order that each
// is dropped.

const PREF_INDENT: u32 = 16;

pub mod d {
    #![allow(unused_parens)]
    use std::fmt;
    use std::mem;
    use std::cell::RefCell;

    static mut counter: u32 = 0;
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
        name: &'static str, i: u32, uid: u32, trail: u32, log: Log<'a>
    }

    impl<'a> fmt::Display for D<'a> {
        fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result {
            write!(w, "D({}_{}, {})", self.name, self.i, self.uid)
        }
    }

    impl<'a> D<'a> {
        pub fn new(name: &'static str, i: u32, log: Log<'a>) -> D<'a> {
            unsafe {
                let trail = first_avail();
                let ctr = counter;
                counter += 1;
                trails |= (1 << trail);
                let ret = D {
                    name: name, i: i, log: log, uid: ctr, trail: trail
                };
                indent_println(trail, &format!("+-- Make {}", ret));
                ret
            }
        }
        pub fn incr(&self) -> D<'a> {
            D::new(self.name, self.i + 1, self.log)
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
