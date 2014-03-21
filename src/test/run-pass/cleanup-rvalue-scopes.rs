// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that destructors for rvalue temporaries run either at end of
// statement or end of block, as appropriate given the temporary
// lifetime rules.

#[feature(macro_rules)];

use std::ops::Drop;

static mut FLAGS: u64 = 0;

struct Box<T> { f: T }
struct AddFlags { bits: u64 }

fn AddFlags(bits: u64) -> AddFlags {
    AddFlags { bits: bits }
}

fn arg(exp: u64, _x: &AddFlags) {
    check_flags(exp);
}

fn pass<T>(v: T) -> T {
    v
}

fn check_flags(exp: u64) {
    unsafe {
        let x = FLAGS;
        FLAGS = 0;
        println!("flags {}, expected {}", x, exp);
        assert_eq!(x, exp);
    }
}

impl AddFlags {
    fn check_flags<'a>(&'a self, exp: u64) -> &'a AddFlags {
        check_flags(exp);
        self
    }

    fn bits(&self) -> u64 {
        self.bits
    }
}

impl Drop for AddFlags {
    fn drop(&mut self) {
        unsafe {
            FLAGS = FLAGS + self.bits;
        }
    }
}

macro_rules! end_of_block(
    ($pat:pat, $expr:expr) => (
        {
            println!("end_of_block({})", stringify!({let $pat = $expr;}));

            {
                // Destructor here does not run until exit from the block.
                let $pat = $expr;
                check_flags(0);
            }
            check_flags(1);
        }
    )
)

macro_rules! end_of_stmt(
    ($pat:pat, $expr:expr) => (
        {
            println!("end_of_stmt({})", stringify!($expr));

            {
                // Destructor here run after `let` statement
                // terminates.
                let $pat = $expr;
                check_flags(1);
            }

            check_flags(0);
        }
    )
)

pub fn main() {

    // In all these cases, we trip over the rules designed to cover
    // the case where we are taking addr of rvalue and storing that
    // addr into a stack slot, either via `let ref` or via a `&` in
    // the initializer.

    end_of_block!(_x, AddFlags(1));
    end_of_block!(_x, &AddFlags(1));
    end_of_block!(_x, & &AddFlags(1));
    end_of_block!(_x, Box { f: AddFlags(1) });
    end_of_block!(_x, Box { f: &AddFlags(1) });
    end_of_block!(_x, Box { f: &AddFlags(1) });
    end_of_block!(_x, pass(AddFlags(1)));
    end_of_block!(ref _x, AddFlags(1));
    end_of_block!(AddFlags { bits: ref _x }, AddFlags(1));
    end_of_block!(&AddFlags { bits }, &AddFlags(1));
    end_of_block!((_, ref _y), (AddFlags(1), 22));
    end_of_block!(~ref _x, ~AddFlags(1));
    end_of_block!(~_x, ~AddFlags(1));
    end_of_block!(_, { { check_flags(0); &AddFlags(1) } });
    end_of_block!(_, &((Box { f: AddFlags(1) }).f));
    end_of_block!(_, &(([AddFlags(1)])[0]));

    // LHS does not create a ref binding, so temporary lives as long
    // as statement, and we do not move the AddFlags out:
    end_of_stmt!(_, AddFlags(1));
    end_of_stmt!((_, _), (AddFlags(1), 22));

    // `&` operator appears inside an arg to a function,
    // so it is not prolonged:
    end_of_stmt!(ref _x, arg(0, &AddFlags(1)));

    // autoref occurs inside receiver, so temp lifetime is not
    // prolonged:
    end_of_stmt!(ref _x, AddFlags(1).check_flags(0).bits());

    // No reference is created on LHS, thus RHS is moved into
    // a temporary that lives just as long as the statement.
    end_of_stmt!(AddFlags { bits }, AddFlags(1));
}
