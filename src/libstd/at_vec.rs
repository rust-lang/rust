// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Managed vectors

use cast::transmute;
use container::Container;
use iterator::IteratorUtil;
use kinds::Copy;
use option::Option;
use sys;
use uint;
use vec::{ImmutableVector, OwnedVector};

/// Code for dealing with @-vectors. This is pretty incomplete, and
/// contains a bunch of duplication from the code for ~-vectors.

pub mod rustrt {
    use libc;
    use vec;
    #[cfg(stage0)]
    use intrinsic::{TyDesc};
    #[cfg(not(stage0))]
    use unstable::intrinsics::{TyDesc};

    #[abi = "cdecl"]
    #[link_name = "rustrt"]
    pub extern {
        pub unsafe fn vec_reserve_shared_actual(t: *TyDesc,
                                                v: **vec::raw::VecRepr,
                                                n: libc::size_t);
    }
}

/// Returns the number of elements the vector can hold without reallocating
#[inline]
pub fn capacity<T>(v: @[T]) -> uint {
    unsafe {
        let repr: **raw::VecRepr = transmute(&v);
        (**repr).unboxed.alloc / sys::size_of::<T>()
    }
}

/**
 * Builds a vector by calling a provided function with an argument
 * function that pushes an element to the back of a vector.
 * This version takes an initial size for the vector.
 *
 * # Arguments
 *
 * * size - An initial size of the vector to reserve
 * * builder - A function that will construct the vector. It receives
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline]
pub fn build_sized<A>(size: uint, builder: &fn(push: &fn(v: A))) -> @[A] {
    let mut vec: @[A] = @[];
    unsafe { raw::reserve(&mut vec, size); }
    builder(|x| unsafe { raw::push(&mut vec, x) });
    return unsafe { transmute(vec) };
}

/**
 * Builds a vector by calling a provided function with an argument
 * function that pushes an element to the back of a vector.
 *
 * # Arguments
 *
 * * builder - A function that will construct the vector. It receives
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline]
pub fn build<A>(builder: &fn(push: &fn(v: A))) -> @[A] {
    build_sized(4, builder)
}

/**
 * Builds a vector by calling a provided function with an argument
 * function that pushes an element to the back of a vector.
 * This version takes an initial size for the vector.
 *
 * # Arguments
 *
 * * size - An option, maybe containing initial size of the vector to reserve
 * * builder - A function that will construct the vector. It receives
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline]
pub fn build_sized_opt<A>(size: Option<uint>,
                          builder: &fn(push: &fn(v: A)))
                       -> @[A] {
    build_sized(size.get_or_default(4), builder)
}

// Appending

/// Iterates over the `rhs` vector, copying each element and appending it to the
/// `lhs`. Afterwards, the `lhs` is then returned for use again.
#[inline]
pub fn append<T:Copy>(lhs: @[T], rhs: &[T]) -> @[T] {
    do build_sized(lhs.len() + rhs.len()) |push| {
        for lhs.iter().advance |x| { push(copy *x); }
        for uint::range(0, rhs.len()) |i| { push(copy rhs[i]); }
    }
}


/// Apply a function to each element of a vector and return the results
pub fn map<T, U>(v: &[T], f: &fn(x: &T) -> U) -> @[U] {
    do build_sized(v.len()) |push| {
        for v.iter().advance |elem| {
            push(f(elem));
        }
    }
}

/**
 * Creates and initializes an immutable vector.
 *
 * Creates an immutable vector of size `n_elts` and initializes the elements
 * to the value returned by the function `op`.
 */
pub fn from_fn<T>(n_elts: uint, op: &fn(uint) -> T) -> @[T] {
    do build_sized(n_elts) |push| {
        let mut i: uint = 0u;
        while i < n_elts { push(op(i)); i += 1u; }
    }
}

/**
 * Creates and initializes an immutable vector.
 *
 * Creates an immutable vector of size `n_elts` and initializes the elements
 * to the value `t`.
 */
pub fn from_elem<T:Copy>(n_elts: uint, t: T) -> @[T] {
    do build_sized(n_elts) |push| {
        let mut i: uint = 0u;
        while i < n_elts { push(copy t); i += 1u; }
    }
}

/**
 * Creates and initializes an immutable managed vector by moving all the
 * elements from an owned vector.
 */
pub fn to_managed_consume<T>(v: ~[T]) -> @[T] {
    let mut av = @[];
    unsafe {
        raw::reserve(&mut av, v.len());
        for v.consume_iter().advance |x| {
            raw::push(&mut av, x);
        }
        transmute(av)
    }
}

/**
 * Creates and initializes an immutable managed vector by copying all the
 * elements of a slice.
 */
pub fn to_managed<T:Copy>(v: &[T]) -> @[T] {
    from_fn(v.len(), |i| copy v[i])
}

#[cfg(not(test))]
pub mod traits {
    use at_vec::append;
    use vec::Vector;
    use kinds::Copy;
    use ops::Add;

    impl<'self,T:Copy, V: Vector<T>> Add<V,@[T]> for @[T] {
        #[inline]
        fn add(&self, rhs: &V) -> @[T] {
            append(*self, rhs.as_slice())
        }
    }
}

#[cfg(test)]
pub mod traits {}

pub mod raw {
    use at_vec::{capacity, rustrt};
    use cast::{transmute, transmute_copy};
    use libc;
    use ptr;
    use sys;
    use uint;
    use unstable::intrinsics::{move_val_init};
    use vec;
    #[cfg(stage0)]
    use intrinsic::{get_tydesc};
    #[cfg(not(stage0))]
    use unstable::intrinsics::{get_tydesc};

    pub type VecRepr = vec::raw::VecRepr;
    pub type SliceRepr = vec::raw::SliceRepr;

    /**
     * Sets the length of a vector
     *
     * This will explicitly set the size of the vector, without actually
     * modifing its buffers, so it is up to the caller to ensure that
     * the vector is actually the specified size.
     */
    #[inline]
    pub unsafe fn set_len<T>(v: @[T], new_len: uint) {
        let repr: **mut VecRepr = transmute(&v);
        (**repr).unboxed.fill = new_len * sys::size_of::<T>();
    }

    /**
     * Pushes a new value onto this vector.
     */
    #[inline]
    pub unsafe fn push<T>(v: &mut @[T], initval: T) {
        let repr: **VecRepr = transmute_copy(&v);
        let fill = (**repr).unboxed.fill;
        if (**repr).unboxed.alloc > fill {
            push_fast(v, initval);
        } else {
            push_slow(v, initval);
        }
    }

    #[inline] // really pretty please
    unsafe fn push_fast<T>(v: &mut @[T], initval: T) {
        let repr: **mut VecRepr = ::cast::transmute(v);
        let fill = (**repr).unboxed.fill;
        (**repr).unboxed.fill += sys::size_of::<T>();
        let p = &((**repr).unboxed.data);
        let p = ptr::offset(p, fill) as *mut T;
        move_val_init(&mut(*p), initval);
    }

    unsafe fn push_slow<T>(v: &mut @[T], initval: T) {
        reserve_at_least(&mut *v, v.len() + 1u);
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
    pub unsafe fn reserve<T>(v: &mut @[T], n: uint) {
        // Only make the (slow) call into the runtime if we have to
        if capacity(*v) < n {
            let ptr: **VecRepr = transmute(v);
            rustrt::vec_reserve_shared_actual(get_tydesc::<T>(),
                                              ptr, n as libc::size_t);
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
    pub unsafe fn reserve_at_least<T>(v: &mut @[T], n: uint) {
        reserve(v, uint::next_power_of_two(n));
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use uint;

    #[test]
    fn test() {
        // Some code that could use that, then:
        fn seq_range(lo: uint, hi: uint) -> @[uint] {
            do build |push| {
                for uint::range(lo, hi) |i| {
                    push(i);
                }
            }
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
    fn test_to_managed_consume() {
        assert_eq!(to_managed_consume::<int>(~[]), @[]);
        assert_eq!(to_managed_consume(~[true]), @[true]);
        assert_eq!(to_managed_consume(~[1, 2, 3, 4, 5]), @[1, 2, 3, 4, 5]);
        assert_eq!(to_managed_consume(~[~"abc", ~"123"]), @[~"abc", ~"123"]);
        assert_eq!(to_managed_consume(~[~[42]]), @[~[42]]);
    }

    #[test]
    fn test_to_managed() {
        assert_eq!(to_managed::<int>([]), @[]);
        assert_eq!(to_managed([true]), @[true]);
        assert_eq!(to_managed([1, 2, 3, 4, 5]), @[1, 2, 3, 4, 5]);
        assert_eq!(to_managed([@"abc", @"123"]), @[@"abc", @"123"]);
        assert_eq!(to_managed([@[42]]), @[@[42]]);
    }
}
