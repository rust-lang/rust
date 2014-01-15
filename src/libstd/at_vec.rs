// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations on managed vectors (`@[T]` type)

use clone::Clone;
use container::Container;
use iter::{Iterator, FromIterator};
use option::{Option, Some, None};
use mem;
use unstable::raw::Repr;
use vec::{ImmutableVector, OwnedVector};

/// Code for dealing with @-vectors. This is pretty incomplete, and
/// contains a bunch of duplication from the code for ~-vectors.

/// Returns the number of elements the vector can hold without reallocating
#[inline]
pub fn capacity<T>(v: @[T]) -> uint {
    unsafe {
        let managed_box = v.repr();
        (*managed_box).data.alloc / mem::size_of::<T>()
    }
}

/**
 * Builds a vector by calling a provided function with an argument
 * function that pushes an element to the back of a vector.
 * The initial size for the vector may optionally be specified
 *
 * # Arguments
 *
 * * size - An option, maybe containing initial size of the vector to reserve
 * * builder - A function that will construct the vector. It receives
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline]
pub fn build<A>(size: Option<uint>, builder: |push: |v: A||) -> @[A] {
    let mut vec = @[];
    unsafe { raw::reserve(&mut vec, size.unwrap_or(4)); }
    builder(|x| unsafe { raw::push(&mut vec, x) });
    vec
}

// Appending

/// Iterates over the `rhs` vector, copying each element and appending it to the
/// `lhs`. Afterwards, the `lhs` is then returned for use again.
#[inline]
pub fn append<T:Clone>(lhs: @[T], rhs: &[T]) -> @[T] {
    build(Some(lhs.len() + rhs.len()), |push| {
        for x in lhs.iter() {
            push((*x).clone());
        }
        for elt in rhs.iter() {
            push(elt.clone());
        }
    })
}


/// Apply a function to each element of a vector and return the results
#[inline]
pub fn map<T, U>(v: &[T], f: |x: &T| -> U) -> @[U] {
    build(Some(v.len()), |push| {
        for elem in v.iter() {
            push(f(elem));
        }
    })
}

/**
 * Creates and initializes an immutable vector.
 *
 * Creates an immutable vector of size `n_elts` and initializes the elements
 * to the value returned by the function `op`.
 */
#[inline]
pub fn from_fn<T>(n_elts: uint, op: |uint| -> T) -> @[T] {
    build(Some(n_elts), |push| {
        let mut i: uint = 0u;
        while i < n_elts { push(op(i)); i += 1u; }
    })
}

/**
 * Creates and initializes an immutable vector.
 *
 * Creates an immutable vector of size `n_elts` and initializes the elements
 * to the value `t`.
 */
#[inline]
pub fn from_elem<T:Clone>(n_elts: uint, t: T) -> @[T] {
    build(Some(n_elts), |push| {
        let mut i: uint = 0u;
        while i < n_elts {
            push(t.clone());
            i += 1u;
        }
    })
}

/**
 * Creates and initializes an immutable managed vector by moving all the
 * elements from an owned vector.
 */
#[inline]
pub fn to_managed_move<T>(v: ~[T]) -> @[T] {
    let mut av = @[];
    unsafe {
        raw::reserve(&mut av, v.len());
        for x in v.move_iter() {
            raw::push(&mut av, x);
        }
        av
    }
}

/**
 * Creates and initializes an immutable managed vector by copying all the
 * elements of a slice.
 */
#[inline]
pub fn to_managed<T:Clone>(v: &[T]) -> @[T] {
    from_fn(v.len(), |i| v[i].clone())
}

impl<T> Clone for @[T] {
    fn clone(&self) -> @[T] {
        *self
    }
}

impl<A> FromIterator<A> for @[A] {
    #[inline]
    fn from_iterator<T: Iterator<A>>(iterator: &mut T) -> @[A] {
        let (lower, _) = iterator.size_hint();
        build(Some(lower), |push| {
            for x in *iterator {
                push(x);
            }
        })
    }
}

#[cfg(not(test))]
#[allow(missing_doc)]
pub mod traits {
    use at_vec::append;
    use clone::Clone;
    use ops::Add;
    use vec::Vector;

    impl<'a,T:Clone, V: Vector<T>> Add<V,@[T]> for @[T] {
        #[inline]
        fn add(&self, rhs: &V) -> @[T] {
            append(*self, rhs.as_slice())
        }
    }
}

#[cfg(test)]
pub mod traits {}

#[allow(missing_doc)]
pub mod raw {
    use at_vec::capacity;
    use cast;
    use cast::{transmute, transmute_copy};
    use container::Container;
    use option::None;
    use ptr;
    use mem;
    use uint;
    use unstable::intrinsics::{move_val_init, TyDesc};
    use unstable::intrinsics;
    use unstable::raw::{Box, Vec};

    /**
     * Sets the length of a vector
     *
     * This will explicitly set the size of the vector, without actually
     * modifying its buffers, so it is up to the caller to ensure that
     * the vector is actually the specified size.
     */
    #[inline]
    pub unsafe fn set_len<T>(v: &mut @[T], new_len: uint) {
        let repr: *mut Box<Vec<T>> = cast::transmute_copy(v);
        (*repr).data.fill = new_len * mem::size_of::<T>();
    }

    /**
     * Pushes a new value onto this vector.
     */
    #[inline]
    pub unsafe fn push<T>(v: &mut @[T], initval: T) {
        let full = {
            let repr: *Box<Vec<T>> = cast::transmute_copy(v);
            (*repr).data.alloc > (*repr).data.fill
        };
        if full {
            push_fast(v, initval);
        } else {
            push_slow(v, initval);
        }
    }

    #[inline] // really pretty please
    unsafe fn push_fast<T>(v: &mut @[T], initval: T) {
        let repr: *mut Box<Vec<T>> = cast::transmute_copy(v);
        let amt = v.len();
        (*repr).data.fill += mem::size_of::<T>();
        let p = ptr::offset(&(*repr).data.data as *T, amt as int) as *mut T;
        move_val_init(&mut(*p), initval);
    }

    #[inline]
    unsafe fn push_slow<T>(v: &mut @[T], initval: T) {
        reserve_at_least(v, v.len() + 1u);
        push_fast(v, initval);
    }

    /**
     * Reserves capacity for exactly `n` elements in the given vector.
     *
     * If the capacity for `v` is already equal to or greater than the
     * requested capacity, then no action is taken.
     *
     * # Arguments
     *
     * * v - A vector
     * * n - The number of elements to reserve space for
     */
    #[inline]
    pub unsafe fn reserve<T>(v: &mut @[T], n: uint) {
        // Only make the (slow) call into the runtime if we have to
        if capacity(*v) < n {
            let ptr: *mut *mut Box<Vec<()>> = transmute(v);
            let ty = intrinsics::get_tydesc::<T>();
            return reserve_raw(ty, ptr, n);
        }
    }

    // Implementation detail. Shouldn't be public
    #[allow(missing_doc)]
    #[inline]
    pub fn reserve_raw(ty: *TyDesc, ptr: *mut *mut Box<Vec<()>>, n: uint) {
        // check for `uint` overflow
        unsafe {
            if n > (**ptr).data.alloc / (*ty).size {
                let alloc = n * (*ty).size;
                let total_size = alloc + mem::size_of::<Vec<()>>();
                if alloc / (*ty).size != n || total_size < alloc {
                    fail!("vector size is too large: {}", n);
                }
                (*ptr) = local_realloc(*ptr as *(), total_size) as *mut Box<Vec<()>>;
                (**ptr).data.alloc = alloc;
            }
        }

        #[inline]
        fn local_realloc(ptr: *(), size: uint) -> *() {
            use rt::local::Local;
            use rt::task::Task;

            let mut task = Local::borrow(None::<Task>);
            task.get().heap.realloc(ptr as *mut Box<()>, size) as *()
        }
    }

    /**
     * Reserves capacity for at least `n` elements in the given vector.
     *
     * This function will over-allocate in order to amortize the
     * allocation costs in scenarios where the caller may need to
     * repeatedly reserve additional space.
     *
     * If the capacity for `v` is already equal to or greater than the
     * requested capacity, then no action is taken.
     *
     * # Arguments
     *
     * * v - A vector
     * * n - The number of elements to reserve space for
     */
    #[inline]
    pub unsafe fn reserve_at_least<T>(v: &mut @[T], n: uint) {
        reserve(v, uint::next_power_of_two(n));
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use prelude::*;
    use bh = extra::test::BenchHarness;

    #[test]
    fn test() {
        // Some code that could use that, then:
        fn seq_range(lo: uint, hi: uint) -> @[uint] {
            build(None, |push| {
                for i in range(lo, hi) {
                    push(i);
                }
            })
        }

        assert_eq!(seq_range(10, 15), @[10, 11, 12, 13, 14]);
        assert_eq!(from_fn(5, |x| x+1), @[1, 2, 3, 4, 5]);
        assert_eq!(from_elem(5, 3.14), @[3.14, 3.14, 3.14, 3.14, 3.14]);
    }

    #[test]
    fn append_test() {
        assert_eq!(@[1,2,3] + &[4,5,6], @[1,2,3,4,5,6]);
    }

    #[test]
    fn test_to_managed_move() {
        assert_eq!(to_managed_move::<int>(~[]), @[]);
        assert_eq!(to_managed_move(~[true]), @[true]);
        assert_eq!(to_managed_move(~[1, 2, 3, 4, 5]), @[1, 2, 3, 4, 5]);
        assert_eq!(to_managed_move(~[~"abc", ~"123"]), @[~"abc", ~"123"]);
        assert_eq!(to_managed_move(~[~[42]]), @[~[42]]);
    }

    #[test]
    fn test_to_managed() {
        assert_eq!(to_managed::<int>([]), @[]);
        assert_eq!(to_managed([true]), @[true]);
        assert_eq!(to_managed([1, 2, 3, 4, 5]), @[1, 2, 3, 4, 5]);
        assert_eq!(to_managed([@"abc", @"123"]), @[@"abc", @"123"]);
        assert_eq!(to_managed([@[42]]), @[@[42]]);
    }

    #[bench]
    fn bench_capacity(b: &mut bh) {
        let x = @[1, 2, 3];
        b.iter(|| {
            let _ = capacity(x);
        });
    }

    #[bench]
    fn bench_build_sized(b: &mut bh) {
        let len = 64;
        b.iter(|| {
            build(Some(len), |push| for i in range(0, 1024) { push(i) });
        });
    }

    #[bench]
    fn bench_build(b: &mut bh) {
        b.iter(|| {
            for i in range(0, 95) {
                build(None, |push| push(i));
            }
        });
    }

    #[bench]
    fn bench_append(b: &mut bh) {
        let lhs = @[7, ..128];
        let rhs = range(0, 256).to_owned_vec();
        b.iter(|| {
            let _ = append(lhs, rhs);
        })
    }

    #[bench]
    fn bench_map(b: &mut bh) {
        let elts = range(0, 256).to_owned_vec();
        b.iter(|| {
            let _ = map(elts, |x| x*2);
        })
    }

    #[bench]
    fn bench_from_fn(b: &mut bh) {
        b.iter(|| {
            let _ = from_fn(1024, |x| x);
        });
    }

    #[bench]
    fn bench_from_elem(b: &mut bh) {
        b.iter(|| {
            let _ = from_elem(1024, 0u64);
        });
    }

    #[bench]
    fn bench_to_managed_move(b: &mut bh) {
        b.iter(|| {
            let elts = range(0, 1024).to_owned_vec(); // yikes! can't move out of capture, though
            to_managed_move(elts);
        })
    }

    #[bench]
    fn bench_to_managed(b: &mut bh) {
        let elts = range(0, 1024).to_owned_vec();
        b.iter(|| {
            let _ = to_managed(elts);
        });
    }

    #[bench]
    fn bench_clone(b: &mut bh) {
        let elts = to_managed(range(0, 1024).to_owned_vec());
        b.iter(|| {
            let _ = elts.clone();
        });
    }
}
