// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Basic functions for dealing with memory.
//!
//! This module contains functions for querying the size and alignment of
//! types, initializing and manipulating memory.

#![stable(feature = "rust1", since = "1.0.0")]

use clone;
use cmp;
use fmt;
use hash;
use intrinsics;
use marker::{Copy, PhantomData, Sized};
use ptr;

#[stable(feature = "rust1", since = "1.0.0")]
pub use intrinsics::transmute;

/// Leaks a value: takes ownership and "forgets" about the value **without running
/// its destructor**.
///
/// Any resources the value manages, such as heap memory or a file handle, will linger
/// forever in an unreachable state.
///
/// If you want to dispose of a value properly, running its destructor, see
/// [`mem::drop`][drop].
///
/// # Safety
///
/// `forget` is not marked as `unsafe`, because Rust's safety guarantees
/// do not include a guarantee that destructors will always run. For example,
/// a program can create a reference cycle using [`Rc`][rc], or call
/// [`process:exit`][exit] to exit without running destructors. Thus, allowing
/// `mem::forget` from safe code does not fundamentally change Rust's safety
/// guarantees.
///
/// That said, leaking resources such as memory or I/O objects is usually undesirable,
/// so `forget` is only recommended for specialized use cases like those shown below.
///
/// Because forgetting a value is allowed, any `unsafe` code you write must
/// allow for this possibility. You cannot return a value and expect that the
/// caller will necessarily run the value's destructor.
///
/// [rc]: ../../std/rc/struct.Rc.html
/// [exit]: ../../std/process/fn.exit.html
///
/// # Examples
///
/// Leak some heap memory by never deallocating it:
///
/// ```
/// use std::mem;
///
/// let heap_memory = Box::new(3);
/// mem::forget(heap_memory);
/// ```
///
/// Leak an I/O object, never closing the file:
///
/// ```no_run
/// use std::mem;
/// use std::fs::File;
///
/// let file = File::open("foo.txt").unwrap();
/// mem::forget(file);
/// ```
///
/// The practical use cases for `forget` are rather specialized and mainly come
/// up in unsafe or FFI code.
///
/// ## Use case 1
///
/// You have created an uninitialized value using [`mem::uninitialized`][uninit].
/// You must either initialize or `forget` it on every computation path before
/// Rust drops it automatically, like at the end of a scope or after a panic.
/// Running the destructor on an uninitialized value would be [undefined behavior][ub].
///
/// ```
/// use std::mem;
/// use std::ptr;
///
/// # let some_condition = false;
/// unsafe {
///     let mut uninit_vec: Vec<u32> = mem::uninitialized();
///
///     if some_condition {
///         // Initialize the variable.
///         ptr::write(&mut uninit_vec, Vec::new());
///     } else {
///         // Forget the uninitialized value so its destructor doesn't run.
///         mem::forget(uninit_vec);
///     }
/// }
/// ```
///
/// ## Use case 2
///
/// You have duplicated the bytes making up a value, without doing a proper
/// [`Clone`][clone]. You need the value's destructor to run only once,
/// because a double `free` is undefined behavior.
///
/// An example is the (old) definition of [`mem::swap`][swap] in this module:
///
/// ```
/// use std::mem;
/// use std::ptr;
///
/// # #[allow(dead_code)]
/// fn swap<T>(x: &mut T, y: &mut T) {
///     unsafe {
///         // Give ourselves some scratch space to work with
///         let mut t: T = mem::uninitialized();
///
///         // Perform the swap, `&mut` pointers never alias
///         ptr::copy_nonoverlapping(&*x, &mut t, 1);
///         ptr::copy_nonoverlapping(&*y, x, 1);
///         ptr::copy_nonoverlapping(&t, y, 1);
///
///         // y and t now point to the same thing, but we need to completely
///         // forget `t` because we do not want to run the destructor for `T`
///         // on its value, which is still owned somewhere outside this function.
///         mem::forget(t);
///     }
/// }
/// ```
///
/// ## Use case 3
///
/// You are transferring ownership across a [FFI] boundary to code written in
/// another language. You need to `forget` the value on the Rust side because Rust
/// code is no longer responsible for it.
///
/// ```no_run
/// use std::mem;
///
/// extern "C" {
///     fn my_c_function(x: *const u32);
/// }
///
/// let x: Box<u32> = Box::new(3);
///
/// // Transfer ownership into C code.
/// unsafe {
///     my_c_function(&*x);
/// }
/// mem::forget(x);
/// ```
///
/// In this case, C code must call back into Rust to free the object. Calling C's `free`
/// function on a [`Box`][box] is *not* safe! Also, `Box` provides an [`into_raw`][into_raw]
/// method which is the preferred way to do this in practice.
///
/// [drop]: fn.drop.html
/// [uninit]: fn.uninitialized.html
/// [clone]: ../clone/trait.Clone.html
/// [swap]: fn.swap.html
/// [FFI]: ../../book/ffi.html
/// [box]: ../../std/boxed/struct.Box.html
/// [into_raw]: ../../std/boxed/struct.Box.html#method.into_raw
/// [ub]: ../../reference/behavior-considered-undefined.html
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn forget<T>(t: T) {
    unsafe { intrinsics::forget(t) }
}

/// Returns the size of a type in bytes.
///
/// More specifically, this is the offset in bytes between successive
/// items of the same type, including alignment padding.
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// assert_eq!(4, mem::size_of::<i32>());
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn size_of<T>() -> usize {
    unsafe { intrinsics::size_of::<T>() }
}

/// Returns the size of the pointed-to value in bytes.
///
/// This is usually the same as `size_of::<T>()`. However, when `T` *has* no
/// statically known size, e.g. a slice [`[T]`][slice] or a [trait object],
/// then `size_of_val` can be used to get the dynamically-known size.
///
/// [slice]: ../../std/primitive.slice.html
/// [trait object]: ../../book/trait-objects.html
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// assert_eq!(4, mem::size_of_val(&5i32));
///
/// let x: [u8; 13] = [0; 13];
/// let y: &[u8] = &x;
/// assert_eq!(13, mem::size_of_val(y));
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn size_of_val<T: ?Sized>(val: &T) -> usize {
    unsafe { intrinsics::size_of_val(val) }
}

/// Returns the [ABI]-required minimum alignment of a type.
///
/// Every valid address of a value of the type `T` must be a multiple of this number.
///
/// This is the alignment used for struct fields. It may be smaller than the preferred alignment.
///
/// [ABI]: https://en.wikipedia.org/wiki/Application_binary_interface
///
/// # Examples
///
/// ```
/// # #![allow(deprecated)]
/// use std::mem;
///
/// assert_eq!(4, mem::min_align_of::<i32>());
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_deprecated(reason = "use `align_of` instead", since = "1.2.0")]
pub fn min_align_of<T>() -> usize {
    unsafe { intrinsics::min_align_of::<T>() }
}

/// Returns the [ABI]-required minimum alignment of the type of the value that `val` points to.
///
/// Every valid address of a value of the type `T` must be a multiple of this number.
///
/// [ABI]: https://en.wikipedia.org/wiki/Application_binary_interface
///
/// # Examples
///
/// ```
/// # #![allow(deprecated)]
/// use std::mem;
///
/// assert_eq!(4, mem::min_align_of_val(&5i32));
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_deprecated(reason = "use `align_of_val` instead", since = "1.2.0")]
pub fn min_align_of_val<T: ?Sized>(val: &T) -> usize {
    unsafe { intrinsics::min_align_of_val(val) }
}

/// Returns the [ABI]-required minimum alignment of a type.
///
/// Every valid address of a value of the type `T` must be a multiple of this number.
///
/// This is the alignment used for struct fields. It may be smaller than the preferred alignment.
///
/// [ABI]: https://en.wikipedia.org/wiki/Application_binary_interface
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// assert_eq!(4, mem::align_of::<i32>());
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn align_of<T>() -> usize {
    unsafe { intrinsics::min_align_of::<T>() }
}

/// Returns the [ABI]-required minimum alignment of the type of the value that `val` points to.
///
/// Every valid address of a value of the type `T` must be a multiple of this number.
///
/// [ABI]: https://en.wikipedia.org/wiki/Application_binary_interface
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// assert_eq!(4, mem::align_of_val(&5i32));
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn align_of_val<T: ?Sized>(val: &T) -> usize {
    unsafe { intrinsics::min_align_of_val(val) }
}

/// Creates a value whose bytes are all zero.
///
/// This has the same effect as allocating space with
/// [`mem::uninitialized`][uninit] and then zeroing it out. It is useful for
/// [FFI] sometimes, but should generally be avoided.
///
/// There is no guarantee that an all-zero byte-pattern represents a valid value of
/// some type `T`. If `T` has a destructor and the value is destroyed (due to
/// a panic or the end of a scope) before being initialized, then the destructor
/// will run on zeroed data, likely leading to [undefined behavior][ub].
///
/// See also the documentation for [`mem::uninitialized`][uninit], which has
/// many of the same caveats.
///
/// [uninit]: fn.uninitialized.html
/// [FFI]: ../../book/ffi.html
/// [ub]: ../../reference/behavior-considered-undefined.html
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// let x: i32 = unsafe { mem::zeroed() };
/// assert_eq!(0, x);
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn zeroed<T>() -> T {
    intrinsics::init()
}

/// Bypasses Rust's normal memory-initialization checks by pretending to
/// produce a value of type `T`, while doing nothing at all.
///
/// **This is incredibly dangerous and should not be done lightly. Deeply
/// consider initializing your memory with a default value instead.**
///
/// This is useful for [FFI] functions and initializing arrays sometimes,
/// but should generally be avoided.
///
/// [FFI]: ../../book/ffi.html
///
/// # Undefined behavior
///
/// It is [undefined behavior][ub] to read uninitialized memory, even just an
/// uninitialized boolean. For instance, if you branch on the value of such
/// a boolean, your program may take one, both, or neither of the branches.
///
/// Writing to the uninitialized value is similarly dangerous. Rust believes the
/// value is initialized, and will therefore try to [`Drop`] the uninitialized
/// value and its fields if you try to overwrite it in a normal manner. The only way
/// to safely initialize an uninitialized value is with [`ptr::write`][write],
/// [`ptr::copy`][copy], or [`ptr::copy_nonoverlapping`][copy_no].
///
/// If the value does implement [`Drop`], it must be initialized before
/// it goes out of scope (and therefore would be dropped). Note that this
/// includes a `panic` occurring and unwinding the stack suddenly.
///
/// # Examples
///
/// Here's how to safely initialize an array of [`Vec`]s.
///
/// ```
/// use std::mem;
/// use std::ptr;
///
/// // Only declare the array. This safely leaves it
/// // uninitialized in a way that Rust will track for us.
/// // However we can't initialize it element-by-element
/// // safely, and we can't use the `[value; 1000]`
/// // constructor because it only works with `Copy` data.
/// let mut data: [Vec<u32>; 1000];
///
/// unsafe {
///     // So we need to do this to initialize it.
///     data = mem::uninitialized();
///
///     // DANGER ZONE: if anything panics or otherwise
///     // incorrectly reads the array here, we will have
///     // Undefined Behavior.
///
///     // It's ok to mutably iterate the data, since this
///     // doesn't involve reading it at all.
///     // (ptr and len are statically known for arrays)
///     for elem in &mut data[..] {
///         // *elem = Vec::new() would try to drop the
///         // uninitialized memory at `elem` -- bad!
///         //
///         // Vec::new doesn't allocate or do really
///         // anything. It's only safe to call here
///         // because we know it won't panic.
///         ptr::write(elem, Vec::new());
///     }
///
///     // SAFE ZONE: everything is initialized.
/// }
///
/// println!("{:?}", &data[0]);
/// ```
///
/// This example emphasizes exactly how delicate and dangerous using `mem::uninitialized`
/// can be. Note that the [`vec!`] macro *does* let you initialize every element with a
/// value that is only [`Clone`], so the following is semantically equivalent and
/// vastly less dangerous, as long as you can live with an extra heap
/// allocation:
///
/// ```
/// let data: Vec<Vec<u32>> = vec![Vec::new(); 1000];
/// println!("{:?}", &data[0]);
/// ```
///
/// [`Vec`]: ../../std/vec/struct.Vec.html
/// [`vec!`]: ../../std/macro.vec.html
/// [`Clone`]: ../../std/clone/trait.Clone.html
/// [ub]: ../../reference/behavior-considered-undefined.html
/// [write]: ../ptr/fn.write.html
/// [copy]: ../intrinsics/fn.copy.html
/// [copy_no]: ../intrinsics/fn.copy_nonoverlapping.html
/// [`Drop`]: ../ops/trait.Drop.html
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn uninitialized<T>() -> T {
    intrinsics::uninit()
}

/// Swaps the values at two mutable locations, without deinitializing either one.
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// let mut x = 5;
/// let mut y = 42;
///
/// mem::swap(&mut x, &mut y);
///
/// assert_eq!(42, x);
/// assert_eq!(5, y);
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn swap<T>(x: &mut T, y: &mut T) {
    unsafe {
        // Give ourselves some scratch space to work with
        let mut t: [u8; 16] = uninitialized();

        let x = x as *mut T as *mut u8;
        let y = y as *mut T as *mut u8;
        let t = &mut t as *mut _ as *mut u8;

        // can't use a for loop as the `range` impl calls `mem::swap` recursively
        let len = size_of::<T>() as isize;
        let mut i = 0;
        while i + 16 <= len {
            // Perform the swap 16 bytes at a time, `&mut` pointers never alias
            ptr::copy_nonoverlapping(x.offset(i), t, 16);
            ptr::copy_nonoverlapping(y.offset(i), x.offset(i), 16);
            ptr::copy_nonoverlapping(t, y.offset(i), 16);
            i += 16;
        }
        if i < len {
            // Swap any remaining bytes
            let rem = (len - i) as usize;
            ptr::copy_nonoverlapping(x.offset(i), t, rem);
            ptr::copy_nonoverlapping(y.offset(i), x.offset(i), rem);
            ptr::copy_nonoverlapping(t, y.offset(i), rem);
        }
    }
}

/// Replaces the value at a mutable location with a new one, returning the old value, without
/// deinitializing either one.
///
/// # Examples
///
/// A simple example:
///
/// ```
/// use std::mem;
///
/// let mut v: Vec<i32> = vec![1, 2];
///
/// let old_v = mem::replace(&mut v, vec![3, 4, 5]);
/// assert_eq!(2, old_v.len());
/// assert_eq!(3, v.len());
/// ```
///
/// `replace` allows consumption of a struct field by replacing it with another value.
/// Without `replace` you can run into issues like these:
///
/// ```ignore
/// struct Buffer<T> { buf: Vec<T> }
///
/// impl<T> Buffer<T> {
///     fn get_and_reset(&mut self) -> Vec<T> {
///         // error: cannot move out of dereference of `&mut`-pointer
///         let buf = self.buf;
///         self.buf = Vec::new();
///         buf
///     }
/// }
/// ```
///
/// Note that `T` does not necessarily implement [`Clone`], so it can't even clone and reset
/// `self.buf`. But `replace` can be used to disassociate the original value of `self.buf` from
/// `self`, allowing it to be returned:
///
/// ```
/// # #![allow(dead_code)]
/// use std::mem;
///
/// # struct Buffer<T> { buf: Vec<T> }
/// impl<T> Buffer<T> {
///     fn get_and_reset(&mut self) -> Vec<T> {
///         mem::replace(&mut self.buf, Vec::new())
///     }
/// }
/// ```
///
/// [`Clone`]: ../../std/clone/trait.Clone.html
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn replace<T>(dest: &mut T, mut src: T) -> T {
    swap(dest, &mut src);
    src
}

/// Disposes of a value.
///
/// While this does call the argument's implementation of [`Drop`][drop],
/// it will not release any borrows, as borrows are based on lexical scope.
///
/// This effectively does nothing for
/// [types which implement `Copy`](../../book/ownership.html#copy-types),
/// e.g. integers. Such values are copied and _then_ moved into the function,
/// so the value persists after this function call.
///
/// This function is not magic; it is literally defined as
///
/// ```
/// pub fn drop<T>(_x: T) { }
/// ```
///
/// Because `_x` is moved into the function, it is automatically dropped before
/// the function returns.
///
/// [drop]: ../ops/trait.Drop.html
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// let v = vec![1, 2, 3];
///
/// drop(v); // explicitly drop the vector
/// ```
///
/// Borrows are based on lexical scope, so this produces an error:
///
/// ```ignore
/// let mut v = vec![1, 2, 3];
/// let x = &v[0];
///
/// drop(x); // explicitly drop the reference, but the borrow still exists
///
/// v.push(4); // error: cannot borrow `v` as mutable because it is also
///            // borrowed as immutable
/// ```
///
/// An inner scope is needed to fix this:
///
/// ```
/// let mut v = vec![1, 2, 3];
///
/// {
///     let x = &v[0];
///
///     drop(x); // this is now redundant, as `x` is going out of scope anyway
/// }
///
/// v.push(4); // no problems
/// ```
///
/// Since [`RefCell`] enforces the borrow rules at runtime, `drop` can
/// release a [`RefCell`] borrow:
///
/// ```
/// use std::cell::RefCell;
///
/// let x = RefCell::new(1);
///
/// let mut mutable_borrow = x.borrow_mut();
/// *mutable_borrow = 1;
///
/// drop(mutable_borrow); // relinquish the mutable borrow on this slot
///
/// let borrow = x.borrow();
/// println!("{}", *borrow);
/// ```
///
/// Integers and other types implementing [`Copy`] are unaffected by `drop`.
///
/// ```
/// #[derive(Copy, Clone)]
/// struct Foo(u8);
///
/// let x = 1;
/// let y = Foo(2);
/// drop(x); // a copy of `x` is moved and dropped
/// drop(y); // a copy of `y` is moved and dropped
///
/// println!("x: {}, y: {}", x, y.0); // still available
/// ```
///
/// [`RefCell`]: ../../std/cell/struct.RefCell.html
/// [`Copy`]: ../../std/marker/trait.Copy.html
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn drop<T>(_x: T) { }

/// Interprets `src` as having type `&U`, and then reads `src` without moving
/// the contained value.
///
/// This function will unsafely assume the pointer `src` is valid for
/// [`size_of::<U>()`][size_of] bytes by transmuting `&T` to `&U` and then reading
/// the `&U`. It will also unsafely create a copy of the contained value instead of
/// moving out of `src`.
///
/// It is not a compile-time error if `T` and `U` have different sizes, but it
/// is highly encouraged to only invoke this function where `T` and `U` have the
/// same size. This function triggers [undefined behavior][ub] if `U` is larger than
/// `T`.
///
/// [ub]: ../../reference/behavior-considered-undefined.html
/// [size_of]: fn.size_of.html
///
/// # Examples
///
/// ```
/// use std::mem;
///
/// #[repr(packed)]
/// struct Foo {
///     bar: u8,
/// }
///
/// let foo_slice = [10u8];
///
/// unsafe {
///     // Copy the data from 'foo_slice' and treat it as a 'Foo'
///     let mut foo_struct: Foo = mem::transmute_copy(&foo_slice);
///     assert_eq!(foo_struct.bar, 10);
///
///     // Modify the copied data
///     foo_struct.bar = 20;
///     assert_eq!(foo_struct.bar, 20);
/// }
///
/// // The contents of 'foo_slice' should not have changed
/// assert_eq!(foo_slice, [10]);
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn transmute_copy<T, U>(src: &T) -> U {
    ptr::read(src as *const T as *const U)
}

/// Opaque type representing the discriminant of an enum.
///
/// See the `discriminant` function in this module for more information.
#[unstable(feature = "discriminant_value", reason = "recently added, follows RFC", issue = "24263")]
pub struct Discriminant<T>(u64, PhantomData<*const T>);

// N.B. These trait implementations cannot be derived because we don't want any bounds on T.

#[unstable(feature = "discriminant_value", reason = "recently added, follows RFC", issue = "24263")]
impl<T> Copy for Discriminant<T> {}

#[unstable(feature = "discriminant_value", reason = "recently added, follows RFC", issue = "24263")]
impl<T> clone::Clone for Discriminant<T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[unstable(feature = "discriminant_value", reason = "recently added, follows RFC", issue = "24263")]
impl<T> cmp::PartialEq for Discriminant<T> {
    fn eq(&self, rhs: &Self) -> bool {
        self.0 == rhs.0
    }
}

#[unstable(feature = "discriminant_value", reason = "recently added, follows RFC", issue = "24263")]
impl<T> cmp::Eq for Discriminant<T> {}

#[unstable(feature = "discriminant_value", reason = "recently added, follows RFC", issue = "24263")]
impl<T> hash::Hash for Discriminant<T> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[unstable(feature = "discriminant_value", reason = "recently added, follows RFC", issue = "24263")]
impl<T> fmt::Debug for Discriminant<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_tuple("Discriminant")
           .field(&self.0)
           .finish()
    }
}

/// Returns a value uniquely identifying the enum variant in `v`.
///
/// If `T` is not an enum, calling this function will not result in undefined behavior, but the
/// return value is unspecified.
///
/// # Stability
///
/// The discriminant of an enum variant may change if the enum definition changes. A discriminant
/// of some variant will not change between compilations with the same compiler.
///
/// # Examples
///
/// This can be used to compare enums that carry data, while disregarding
/// the actual data:
///
/// ```
/// #![feature(discriminant_value)]
/// use std::mem;
///
/// enum Foo { A(&'static str), B(i32), C(i32) }
///
/// assert!(mem::discriminant(&Foo::A("bar")) == mem::discriminant(&Foo::A("baz")));
/// assert!(mem::discriminant(&Foo::B(1))     == mem::discriminant(&Foo::B(2)));
/// assert!(mem::discriminant(&Foo::B(3))     != mem::discriminant(&Foo::C(3)));
/// ```
#[unstable(feature = "discriminant_value", reason = "recently added, follows RFC", issue = "24263")]
pub fn discriminant<T>(v: &T) -> Discriminant<T> {
    unsafe {
        Discriminant(intrinsics::discriminant_value(v), PhantomData)
    }
}

