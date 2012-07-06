// Dynamic Vector
//
// A growable vector that makes use of unique pointers so that the
// result can be sent between tasks and so forth.
//
// Note that recursive use is not permitted.

import dvec_iter::extensions;
import unsafe::reinterpret_cast;
import ptr::{null, extensions};

export dvec;
export from_elem;
export from_vec;
export extensions;
export unwrap;

/**
 * A growable, modifiable vector type that accumulates elements into a
 * unique vector.
 *
 * # Limitations on recursive use
 *
 * This class works by swapping the unique vector out of the data
 * structure whenever it is to be used.  Therefore, recursive use is not
 * permitted.  That is, while iterating through a vector, you cannot
 * access the vector in any other way or else the program will fail.  If
 * you wish, you can use the `swap()` method to gain access to the raw
 * vector and transform it or use it any way you like.  Eventually, we
 * may permit read-only access during iteration or other use.
 *
 * # WARNING
 *
 * For maximum performance, this type is implemented using some rather
 * unsafe code.  In particular, this innocent looking `[mut A]/~` pointer
 * *may be null!*  Therefore, it is important you not reach into the
 * data structure manually but instead use the provided extensions.
 *
 * The reason that I did not use an unsafe pointer in the structure
 * itself is that I wanted to ensure that the vector would be freed when
 * the dvec is dropped.  The reason that I did not use an `option<T>`
 * instead of a nullable pointer is that I found experimentally that it
 * becomes approximately 50% slower. This can probably be improved
 * through optimization.  You can run your own experiments using
 * `src/test/bench/vec-append.rs`. My own tests found that using null
 * pointers achieved about 103 million pushes/second.  Using an option
 * type could only produce 47 million pushes/second.
 */
type dvec<A> = {
    mut data: ~[mut A]
};

/// Creates a new, empty dvec
fn dvec<A>() -> dvec<A> {
    {mut data: ~[mut]}
}

/// Creates a new dvec with a single element
fn from_elem<A>(+e: A) -> dvec<A> {
    {mut data: ~[mut e]}
}

/// Creates a new dvec with the contents of a vector
fn from_vec<A>(+v: ~[mut A]) -> dvec<A> {
    {mut data: v}
}

/// Consumes the vector and returns its contents
fn unwrap<A>(-d: dvec<A>) -> ~[mut A] {
    let {data: v} <- d;
    ret v;
}

impl private_methods<A> for dvec<A> {
    fn check_not_borrowed() {
        unsafe {
            let data: *() = unsafe::reinterpret_cast(self.data);
            if data.is_null() {
                fail "Recursive use of dvec";
            }
        }
    }

    #[inline(always)]
    fn borrow<B>(f: fn(-~[mut A]) -> B) -> B {
        unsafe {
            let mut data = unsafe::reinterpret_cast(null::<()>());
            data <-> self.data;
            let data_ptr: *() = unsafe::reinterpret_cast(data);
            if data_ptr.is_null() { fail "Recursive use of dvec"; }
            ret f(data);
        }
    }

    #[inline(always)]
    fn return(-data: ~[mut A]) {
        unsafe {
            self.data <- data;
        }
    }
}

// In theory, most everything should work with any A, but in practice
// almost nothing works without the copy bound due to limitations
// around closures.
impl extensions<A> for dvec<A> {
    /**
     * Swaps out the current vector and hands it off to a user-provided
     * function `f`.  The function should transform it however is desired
     * and return a new vector to replace it with.
     */
    #[inline(always)]
    fn swap(f: fn(-~[mut A]) -> ~[mut A]) {
        self.borrow(|v| self.return(f(v)))
    }

    /// Returns the number of elements currently in the dvec
    fn len() -> uint {
        do self.borrow |v| {
            let l = v.len();
            self.return(v);
            l
        }
    }

    /// Overwrite the current contents
    fn set(+w: ~[mut A]) {
        self.check_not_borrowed();
        self.data <- w;
    }

    /// Remove and return the last element
    fn pop() -> A {
        do self.borrow |v| {
            let mut v <- v;
            let result = vec::pop(v);
            self.return(v);
            result
        }
    }

    /// Insert a single item at the front of the list
    fn unshift(-t: A) {
        unsafe {
            let mut data = unsafe::reinterpret_cast(null::<()>());
            data <-> self.data;
            let data_ptr: *() = unsafe::reinterpret_cast(data);
            if data_ptr.is_null() { fail "Recursive use of dvec"; }
            log(error, "a");
            self.data <- ~[mut t];
            vec::push_all_move(self.data, data);
            log(error, "b");
        }
    }

    /// Append a single item to the end of the list
    fn push(+t: A) {
        self.check_not_borrowed();
        vec::push(self.data, t);
    }

    /// Remove and return the first element
    fn shift() -> A {
        do self.borrow |v| {
            let mut v = vec::from_mut(v);
            let result = vec::shift(v);
            self.return(vec::to_mut(v));
            result
        }
    }
}

impl extensions<A:copy> for dvec<A> {
    /**
     * Append all elements of a vector to the end of the list
     *
     * Equivalent to `append_iter()` but potentially more efficient.
     */
    fn push_all(ts: &[const A]) {
        self.push_slice(ts, 0u, vec::len(ts));
    }

    /// Appends elements from `from_idx` to `to_idx` (exclusive)
    fn push_slice(ts: &[const A], from_idx: uint, to_idx: uint) {
        do self.swap |v| {
            let mut v <- v;
            let new_len = vec::len(v) + to_idx - from_idx;
            vec::reserve(v, new_len);
            let mut i = from_idx;
            while i < to_idx {
                vec::push(v, ts[i]);
                i += 1u;
            }
            v
        }
    }

    /*
    /**
     * Append all elements of an iterable.
     *
     * Failure will occur if the iterable's `each()` method
     * attempts to access this vector.
     */
    fn append_iter<A, I:iter::base_iter<A>>(ts: I) {
        do self.swap |v| {
           let mut v = alt ts.size_hint() {
             none { v }
             some(h) {
               let len = v.len() + h;
               let mut v <- v;
               vec::reserve(v, len);
               v
            }
           };

        for ts.each |t| { vec::push(v, t) };
           v
        }
    }
    */

    /**
     * Gets a copy of the current contents.
     *
     * See `unwrap()` if you do not wish to copy the contents.
     */
    fn get() -> ~[A] {
        do self.borrow |v| {
            let w = vec::from_mut(copy v);
            self.return(v);
            w
        }
    }

    /// Copy out an individual element
    #[inline(always)]
    fn [](idx: uint) -> A {
        self.get_elt(idx)
    }

    /// Copy out an individual element
    #[inline(always)]
    fn get_elt(idx: uint) -> A {
        self.check_not_borrowed();
        ret self.data[idx];
    }

    /// Overwrites the contents of the element at `idx` with `a`
    fn set_elt(idx: uint, a: A) {
        self.check_not_borrowed();
        self.data[idx] = a;
    }

    /**
     * Overwrites the contents of the element at `idx` with `a`,
     * growing the vector if necessary.  New elements will be initialized
     * with `initval`
     */
    fn grow_set_elt(idx: uint, initval: A, val: A) {
        do self.swap |v| {
            let mut v <- v;
            vec::grow_set(v, idx, initval, val);
            v
        }
    }

    /// Returns the last element, failing if the vector is empty
    #[inline(always)]
    fn last() -> A {
        self.check_not_borrowed();

        let length = self.len();
        if length == 0u {
            fail "attempt to retrieve the last element of an empty vector";
        }

        ret self.data[length - 1u];
    }

    /// Iterates over the elements in reverse order
    #[inline(always)]
    fn reach(f: fn(A) -> bool) {
        let length = self.len();
        let mut i = 0u;
        while i < length {
            if !f(self.get_elt(i)) {
                break;
            }
            i += 1u;
        }
    }
}
