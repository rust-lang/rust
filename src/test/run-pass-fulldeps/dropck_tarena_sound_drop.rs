#![allow(unknown_lints)]
// Check that an arena (TypedArena) can carry elements whose drop
// methods might access borrowed data, as long as the borrowed data
// has lifetime that strictly outlives the arena itself.
//
// Compare against compile-fail/dropck_tarena_unsound_drop.rs, which
// shows a similar setup, but restricts `f` so that the struct `C<'a>`
// is force-fed a lifetime equal to that of the borrowed arena.

#![allow(unstable)]
#![feature(rustc_private)]

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

struct C<'a> { _v: CheckId<&'a usize>, }

impl<'a> HasId for &'a usize { fn count(&self) -> usize { 1 } }

fn f<'a, 'b>(_arena: &'a TypedArena<C<'b>>) {}

fn main() {
    let arena: TypedArena<C> = TypedArena::default();
    f(&arena);
}
