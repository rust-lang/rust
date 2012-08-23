//! Shared Vectors

import ptr::addr_of;

export init_op;
export capacity;
export build_sized, build, build_sized_opt;
export map;
export from_fn, from_elem;
export unsafe;

/// Code for dealing with @-vectors. This is pretty incomplete, and
/// contains a bunch of duplication from the code for ~-vectors.

#[abi = "cdecl"]
extern mod rustrt {
    fn vec_reserve_shared_actual(++t: *sys::TypeDesc,
                                 ++v: **vec::unsafe::VecRepr,
                                 ++n: libc::size_t);
}

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn move_val_init<T>(&dst: T, -src: T);
}

/// A function used to initialize the elements of a vector
type InitOp<T> = fn(uint) -> T;

/// Returns the number of elements the vector can hold without reallocating
#[inline(always)]
pure fn capacity<T>(&&v: @[const T]) -> uint {
    unsafe {
        let repr: **unsafe::VecRepr =
            ::unsafe::reinterpret_cast(addr_of(v));
        (**repr).alloc / sys::size_of::<T>()
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
 * * builder - A function that will construct the vector. It recieves
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline(always)]
pure fn build_sized<A>(size: uint, builder: fn(push: pure fn(+A))) -> @[A] {
    let mut vec = @[];
    unsafe {
        unsafe::reserve(vec, size);
        // This is an awful hack to be able to make the push function
        // pure. Is there a better way?
        ::unsafe::reinterpret_cast::
            <fn(push: pure fn(+A)), fn(push: fn(+A))>
            (builder)(|+x| unsafe::push(vec, x));
    }
    return vec;
}

/**
 * Builds a vector by calling a provided function with an argument
 * function that pushes an element to the back of a vector.
 *
 * # Arguments
 *
 * * builder - A function that will construct the vector. It recieves
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline(always)]
pure fn build<A>(builder: fn(push: pure fn(+A))) -> @[A] {
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
 * * builder - A function that will construct the vector. It recieves
 *             as an argument a function that will push an element
 *             onto the vector being constructed.
 */
#[inline(always)]
pure fn build_sized_opt<A>(size: option<uint>,
                           builder: fn(push: pure fn(+A))) -> @[A] {
    build_sized(size.get_default(4), builder)
}

// Appending
#[inline(always)]
pure fn append<T: copy>(lhs: @[T], rhs: &[const T]) -> @[T] {
    do build_sized(lhs.len() + rhs.len()) |push| {
        for vec::each(lhs) |x| { push(x); }
        for uint::range(0, rhs.len()) |i| { push(rhs[i]); }
    }
}


/// Apply a function to each element of a vector and return the results
pure fn map<T, U>(v: &[T], f: fn(T) -> U) -> @[U] {
    do build_sized(v.len()) |push| {
        for vec::each(v) |elem| {
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
pure fn from_fn<T>(n_elts: uint, op: InitOp<T>) -> @[T] {
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
pure fn from_elem<T: copy>(n_elts: uint, t: T) -> @[T] {
    do build_sized(n_elts) |push| {
        let mut i: uint = 0u;
        while i < n_elts { push(t); i += 1u; }
    }
}

#[cfg(notest)]
impl<T: copy> @[T]: add<&[const T],@[T]> {
    #[inline(always)]
    pure fn add(rhs: &[const T]) -> @[T] {
        append(self, rhs)
    }
}


mod unsafe {
    type VecRepr = vec::unsafe::VecRepr;
    type SliceRepr = vec::unsafe::SliceRepr;

    /**
     * Sets the length of a vector
     *
     * This will explicitly set the size of the vector, without actually
     * modifing its buffers, so it is up to the caller to ensure that
     * the vector is actually the specified size.
     */
    #[inline(always)]
    unsafe fn set_len<T>(&&v: @[const T], new_len: uint) {
        let repr: **VecRepr = ::unsafe::reinterpret_cast(addr_of(v));
        (**repr).fill = new_len * sys::size_of::<T>();
    }

    #[inline(always)]
    unsafe fn push<T>(&v: @[const T], +initval: T) {
        let repr: **VecRepr = ::unsafe::reinterpret_cast(addr_of(v));
        let fill = (**repr).fill;
        if (**repr).alloc > fill {
            push_fast(v, initval);
        }
        else {
            push_slow(v, initval);
        }
    }
    // This doesn't bother to make sure we have space.
    #[inline(always)] // really pretty please
    unsafe fn push_fast<T>(&v: @[const T], +initval: T) {
        let repr: **VecRepr = ::unsafe::reinterpret_cast(addr_of(v));
        let fill = (**repr).fill;
        (**repr).fill += sys::size_of::<T>();
        let p = ptr::addr_of((**repr).data);
        let p = ptr::offset(p, fill) as *mut T;
        rusti::move_val_init(*p, initval);
    }

    unsafe fn push_slow<T>(&v: @[const T], +initval: T) {
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
    unsafe fn reserve<T>(&v: @[const T], n: uint) {
        // Only make the (slow) call into the runtime if we have to
        if capacity(v) < n {
            let ptr = addr_of(v) as **VecRepr;
            rustrt::vec_reserve_shared_actual(sys::get_type_desc::<T>(),
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
    unsafe fn reserve_at_least<T>(&v: @[const T], n: uint) {
        reserve(v, uint::next_power_of_two(n));
    }

}

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

    assert seq_range(10, 15) == @[10, 11, 12, 13, 14];
    assert from_fn(5, |x| x+1) == @[1, 2, 3, 4, 5];
    assert from_elem(5, 3.14) == @[3.14, 3.14, 3.14, 3.14, 3.14];
}

#[test]
fn append_test() {
    assert @[1,2,3] + @[4,5,6] == @[1,2,3,4,5,6];
}

