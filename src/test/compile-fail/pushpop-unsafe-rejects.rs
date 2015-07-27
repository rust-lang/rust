// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Basic sanity check for `push_unsafe!(EXPR)` and
// `pop_unsafe!(EXPR)`: we can call unsafe code when there are a
// positive number of pushes in the stack, or if we are within a
// normal `unsafe` block, but otherwise cannot.

#![feature(pushpop_unsafe)]

static mut X: i32 = 0;

unsafe fn f() { X += 1; return; }
fn g() { unsafe { X += 1_000; } return; }

fn main() {
    push_unsafe!( {
        f(); pop_unsafe!({
            f() //~ ERROR: call to unsafe function
        })
    } );

    push_unsafe!({
        f();
        pop_unsafe!({
            g();
            f(); //~ ERROR: call to unsafe function
        })
    } );

    push_unsafe!({
        g(); pop_unsafe!({
            unsafe {
                f();
            }
            f(); //~ ERROR: call to unsafe function
        })
    });


    // Note: For implementation simplicity the compiler just
    // ICE's if you underflow the push_unsafe stack.
    //
    // Thus all of the following cases cause an ICE.
    //
    // (The "ERROR" notes are from an earlier version
    //  that used saturated arithmetic rather than checked
    //  arithmetic.)

    //    pop_unsafe!{ g() };
    //
    //    push_unsafe!({
    //        pop_unsafe!(pop_unsafe!{ g() })
    //    });
    //
    //    push_unsafe!({
    //        g();
    //        pop_unsafe!(pop_unsafe!({
    //            f() // ERROR: call to unsafe function
    //        }))
    //    });
    //
    //    pop_unsafe!({
    //        f(); // ERROR: call to unsafe function
    //    })

}
