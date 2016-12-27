// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that drop elaboration clears the "master" discriminant
// drop flag even if it protects no fields.

struct Good(usize);
impl Drop for Good {
    #[inline(never)]
    fn drop(&mut self) {
        println!("dropping Good({})", self.0);
    }
}

struct Void;
impl Drop for Void {
    #[inline(never)]
    fn drop(&mut self) {
        panic!("Suddenly, a Void appears.");
    }
}

enum E {
    Never(Void),
    Fine(Good)
}

fn main() {
    let mut go = true;

    loop {
        let next;
        match go {
            true => next = E::Fine(Good(123)),
            false => return,
        }

        match next {
            E::Never(_) => return,
            E::Fine(_good) => go = false,
        }

        // `next` is dropped and StorageDead'd here. We must reset the
        // discriminant's drop flag to avoid random variants being
        // dropped.
    }
}
