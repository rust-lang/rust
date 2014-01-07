// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Task-local garbage-collected boxes

The `Gc` type provides shared ownership of an immutable value. Destruction is not deterministic, and
will occur some time between every `Gc` handle being gone and the end of the task. The garbage
collector is task-local so `Gc<T>` is not sendable.

*/

#[allow(missing_doc)];
#[allow(experimental)];

use kinds::Freeze;
use clone::{Clone, DeepClone};
use mem;
use option::{Some, None};
use rt::local;
use rt::task::{Task, GcUninit, GcExists, GcBorrowed};
use util::replace;

use unstable::intrinsics::{owns_new_managed, move_val_init};

pub mod collector;

/// Immutable garbage-collected pointer type.
///
/// # Warning
///
/// This is highly experimental. Placing them in the wrong place can
/// cause live pointers to be deallocated or reused. Wrong places
/// can include:
///
/// - global variables (including `#[thread_local]` ones)
/// - task-local storage
/// - both built-in allocating pointer types (`~` and `@`)
/// - both built-in allocating vector types (`~[]` and `@[]`)
/// - most library smart pointers, like `Rc`
#[no_send]
#[experimental]
#[managed]
pub struct Gc<T> {
    priv ptr: *T
}

impl<T: 'static> Gc<T> {
    /// Construct a new garbage-collected box
    #[experimental="not rooted by built-in pointer and vector types"]
    pub fn new(value: T) -> Gc<T> {
        let stack_top;
        let gc;
        {
            // we need the task-local GC, and some upper bound on the
            // top of the stack. The borrow is scoped so that we can
            // use task things like logging etc. inside .collect() and
            // (as much as possible) inside finalisers.
            let mut task_ = local::Local::borrow(None::<Task>);
            let task = task_.get();

            match task.stack_bounds() {
                (_, t) => stack_top = t,
            }

            // mark the GC as borrowed: unfortunately this means that
            // we can't use an GC functions any finalisers,
            // e.g. Gc<Vec<Gc<T>>> will fail/crash when collected.
            gc = replace(&mut task.gc, GcBorrowed);
        }


        let mut gc = match gc {
            // first GC allocation in this task, so create a new
            // collector
            GcUninit => ~collector::GarbageCollector::new(),
            GcExists(gc) => gc,
            GcBorrowed => fail!("Gc::new: Gc already borrowed.")
        };

        let size = mem::size_of::<T>();
        let ptr;
        unsafe {
            gc.occasional_collection(stack_top);

            // if we don't contain anything that contains has a
            // #[managed] pointer unboxed, then we don't don't need to
            // scan, because there can never be a GC reference inside.
            // FIXME: we currently count ~Gc<int> as owning managed,
            // but it shouldn't (~, or equivalent) should root the Gc
            // itself.
            ptr = if owns_new_managed::<T>() {
                gc.alloc_gc(size)
            } else {
                gc.alloc_gc_no_scan(size)
            } as *mut T;

            move_val_init(&mut *ptr, value);
        }

        // restore the garbage collector to the task.
        let mut task = local::Local::borrow(None::<Task>);
        task.get().gc = GcExists(gc);

        Gc { ptr: ptr as *T }
    }
}

impl<T: 'static + Freeze> Gc<T> {
    /// Borrow the value contained in the garbage-collected box.
    ///
    /// This is restricted to deeply immutable values, and so does not
    /// require a write-barrier because no writes are possible.
    ///
    /// Currently `unsafe` because `~` and `~[]` do not root a `Gc<T>`
    /// box, and so, if that is the only reference to one, then that
    /// `Gc<T>` may be deallocated or the memory reused.
    #[inline]
    pub unsafe fn borrow<'r>(&'r self) -> &'r T {
        &*self.ptr
    }
}

impl<T: 'static> Gc<T> {
    /// Borrow the value contained in the garbage-collected box, with
    /// a write-barrier.
    ///
    /// See `.borrow()` for the reason for `unsafe`.
    #[inline]
    pub unsafe fn borrow_write_barrier<'r>(&'r self) -> &'r T {
        // a completely conservative non-generational GC needs no
        // write barriers.
        &*self.ptr
    }

    /// Borrow the value contained in the garbage-collected box,
    /// without a write-barrier.
    ///
    /// Because this has no write barrier, any writes to the value
    /// must not write new references to other garbage-collected box.
    #[inline]
    pub unsafe fn borrow_no_write_barrier<'r>(&'r self) -> &'r T {
        // a completely conservative non-generational GC needs no
        // write barriers.
        &*self.ptr
    }
}

impl<T> Clone for Gc<T> {
    fn clone(&self) -> Gc<T> { *self }
}

/// The `Freeze` bound restricts this to acyclic graphs where it is well-defined.
///
/// A `Send` bound would also work, but `Send` *or* `Freeze` cannot be expressed.
impl<T: DeepClone + 'static> DeepClone for Gc<T> {
    #[inline]
    fn deep_clone(&self) -> Gc<T> {
        Gc::new(unsafe {self.borrow_write_barrier().deep_clone()})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cell::RefCell;
    use iter::Iterator;
    use option::{Some, None};
    use vec::{ImmutableVector, MutableVector};

    #[test]
    fn test_clone() {
        let x = Gc::new(RefCell::new(5));
        let y = x.clone();
        unsafe {
            x.borrow_write_barrier().with_mut(|inner| {
                    *inner = 20;
                });
            assert_eq!(y.borrow_write_barrier().with(|x| *x), 20);
        }
    }

    #[test]
    fn test_deep_clone() {
        let x = Gc::new(RefCell::new(5));
        let y = x.deep_clone();
        unsafe {
            x.borrow_write_barrier().with_mut(|inner| {
                    *inner = 20;
                });
            assert_eq!(y.borrow_write_barrier().with(|x| *x), 5);
        }
    }

    #[test]
    fn test_simple() {
        let x = Gc::new(5);
        unsafe {
            assert_eq!(*x.borrow(), 5);
        }
    }

    #[test]
    fn test_simple_clone() {
        let x = Gc::new(5);
        let y = x.clone();
        unsafe {
            assert_eq!(*x.borrow(), 5);
            assert_eq!(*y.borrow(), 5);
        }
    }

    #[test]
    #[ignore] // no finalisation of GC'd objects
    fn test_destructor() {
        let x = Gc::new(~5);
        unsafe {
            assert_eq!(**x.borrow(), 5);
        }
    }

    #[test]
    fn test_many_allocs() {
        // on the stack.
        let mut ptrs = [None::<Gc<uint>>, .. 10000];
        for (i, ptr) in ptrs.mut_iter().enumerate() {
            *ptr = Some(Gc::new(i))
        }

        for (i, ptr) in ptrs.iter().enumerate() {
            unsafe {
                let p = ptr.unwrap();
                assert_eq!(*p.borrow(), i);
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use iter::Iterator;
    use option::{Some, None};
    use vec::{ImmutableVector, MutableVector};
    use extra::test::BenchHarness;

    #[bench]
    fn many_allocs(bh: &mut BenchHarness) {
        bh.iter(|| {
                let mut ptrs = [None::<Gc<uint>>, .. 1000];
                for (i, ptr) in ptrs.mut_iter().enumerate() {
                    *ptr = Some(Gc::new(i))
                }
            })
    }
}
