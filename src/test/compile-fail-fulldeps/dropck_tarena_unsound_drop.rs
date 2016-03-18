// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that an arena (TypedArena) cannot carry elements whose drop
// methods might access borrowed data of lifetime that does not
// strictly outlive the arena itself.
//
// Compare against run-pass/dropck_tarena_sound_drop.rs, which shows a
// similar setup, but loosens `f` so that the struct `C<'a>` can be
// fed a lifetime longer than that of the arena.
//
// (Also compare against dropck_tarena_cycle_checked.rs, from which
// this was reduced to better understand its error message.)

extern crate arena;

use arena::TypedArena;

trait HasId { fn count(&self) -> usize; }

struct CheckId<T:HasId> { v: T }

// In the code below, the impl of HasId for `&'a usize` does not
// actually access the borrowed data, but the point is that the
// interface to CheckId does not (and cannot) know that, and therefore
// when encountering a value V of type CheckId<S>, we must
// conservatively force the type S to strictly outlive V.
impl<T:HasId> Drop for CheckId<T> {
    fn drop(&mut self) {
        assert!(self.v.count() > 0);
    }
}

struct C<'a> { v: CheckId<&'a usize>, }

impl<'a> HasId for &'a usize { fn count(&self) -> usize { 1 } }

fn f<'a>(_arena: &'a TypedArena<C<'a>>) {}

fn main() {
    let arena: TypedArena<C> = TypedArena::new();
    f(&arena); //~ ERROR `arena` does not live long enough
}
