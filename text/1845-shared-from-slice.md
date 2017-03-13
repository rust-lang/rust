- Feature Name: `shared_from_slice`
- Start Date: 2017-01-05
- RFC PR: [rust-lang/rfcs#1845](https://github.com/rust-lang/rfcs/pull/1845)
- Rust Issue: [rust-lang/rust#40475](https://github.com/rust-lang/rust/issues/40475)

# Summary
[summary]: #summary

This is an RFC to add the APIs: `From<&[T]> for Rc<[T]>` where [`T: Clone`][Clone] or [`T: Copy`][Copy] as well as `From<&str> for Rc<str>`. In addition: `From<Vec<T>> for Rc<[T]>` and `From<Box<T: ?Sized>> for Rc<T>` will be added.

Identical APIs will also be added for [`Arc`][Arc].

# Motivation
[motivation]: #motivation

## Caching and [string interning]

These, and especially the latter - i.e: `From<&str>`, trait implementations of [`From`][From] are useful when dealing with any form of caching of slices.

This especially applies to *controllable* [string interning], where you can cheaply cache strings with a construct such as putting [`Rc`][Rc]s into [`HashSet`][HashSet]s, i.e: `HashSet<Rc<str>>`.

An example of string interning:

```rust
#![feature(ptr_eq)]
#![feature(shared_from_slice)]
use std::rc::Rc;
use std::collections::HashSet;
use std::mem::drop;

fn cache_str(cache: &mut HashSet<Rc<str>>, input: &str) -> Rc<str> {
     // If the input hasn't been cached, do it:
     if !cache.contains(input) {
         cache.insert(input.into());
     }

    // Retrieve the cached element.
    cache.get(input).unwrap().clone()
}

let first   = "hello world!";
let second  = "goodbye!";
let mut set = HashSet::new();

// Cache the slices:
let rc_first  = cache_str(&mut set, first);
let rc_second = cache_str(&mut set, second);
let rc_third  = cache_str(&mut set, second);

// The contents match:
assert_eq!(rc_first.as_ref(),  first);
assert_eq!(rc_second.as_ref(), second);
assert_eq!(rc_third.as_ref(),  rc_second.as_ref());

// It was cached:
assert_eq!(set.len(), 2);
drop(set);
assert_eq!(Rc::strong_count(&rc_first),  1);
assert_eq!(Rc::strong_count(&rc_second), 2);
assert_eq!(Rc::strong_count(&rc_third),  2);
assert!(Rc::ptr_eq(&rc_second, &rc_third));
```

One could imagine a scenario where you have an [AST][Abstract Syntax Tree] with string literals that gets repeated a lot in it. For example, [namespaces][namespace] in [XML] documents tends to be repeated many times.

The [tendril] crate does one form of interning:
> Buffer sharing is accomplished through thread-local (non-atomic) reference counting

It is useful to provide an implementation of `From<&[T]>` as well, and not just for [`&str`][str], because one might deal with non-utf8 strings, i.e: `&[u8]`. One could potentially reuse this for [`Path`][Path], [`OsStr`][OsStr].

## Safe abstraction for `unsafe` code.

Providing these implementations in the current state of Rust requires substantial amount of `unsafe` code. Therefore, for the sake of confidence in that the implementations are safe - it is best done in the standard library.

## [`RcBox`][RcBox] is not public

Furthermore, since [`RcBox`][RcBox] is not exposed publically from [`std::rc`][std::rc], one can't make an implementation outside of the standard library for this without making assumptions about the internal layout of [`Rc`][Rc]. The alternative is to roll your own implementation of [`Rc`][Rc] in its entirity - but this in turn requires using a lot of feature gates, which makes using this on stable Rust in the near future unfeasible.

## For [`Arc`][Arc]

For [`Arc`][Arc] the synchronization overhead of doing `.clone()` is probably greater than the overhead of doing `Arc<Box<str>>`. But once the clones have been made, `Arc<str>` would probably be cheaper to dereference due to locality.

Most of the motivations for [`Rc`][Rc] applies to [`Arc`][Arc] as well, but the use cases might be fewer. Therefore, the case for adding the same API for [`Arc`][Arc] is less clear. One could perhaps use it for multi threaded interning with a type such as: `Arc<Mutex<HashSet<Arc<str>>>>`.

Because of the similarities between the layout of [`Rc`][Rc] and [`Arc`][Arc], almost identical implementations could be added for `From<&[T]> for Arc<[T]>` and `From<&str> for Arc<str>`. It would also be consistent to do so.

Taking all of this into account, adding the APIs for [`Arc`][Arc] is warranted.

# Detailed design
[design]: #detailed-design

## There's already an implementation
[theres-already-an-implementation]: #theres-already-an-implementation

There is [already an implementation](https://doc.rust-lang.org/nightly/src/alloc/rc.rs.html#417-440) of sorts [`alloc::rc`][Rc] for this. But it is hidden under the feature gate `rustc_private`, which, to the authors knowledge, will never be stabilized. The implementation is, on this day, as follows:

```rust
impl Rc<str> {
    /// Constructs a new `Rc<str>` from a string slice.
    #[doc(hidden)]
    #[unstable(feature = "rustc_private",
               reason = "for internal use in rustc",
               issue = "0")]
    pub fn __from_str(value: &str) -> Rc<str> {
        unsafe {
            // Allocate enough space for `RcBox<str>`.
            let aligned_len = 2 + (value.len() + size_of::<usize>() - 1) / size_of::<usize>();
            let vec = RawVec::<usize>::with_capacity(aligned_len);
            let ptr = vec.ptr();
            forget(vec);
            // Initialize fields of `RcBox<str>`.
            *ptr.offset(0) = 1; // strong: Cell::new(1)
            *ptr.offset(1) = 1; // weak: Cell::new(1)
            ptr::copy_nonoverlapping(value.as_ptr(), ptr.offset(2) as *mut u8, value.len());
            // Combine the allocation address and the string length into a fat pointer to `RcBox`.
            let rcbox_ptr: *mut RcBox<str> = mem::transmute([ptr as usize, value.len()]);
            assert!(aligned_len * size_of::<usize>() == size_of_val(&*rcbox_ptr));
            Rc { ptr: Shared::new(rcbox_ptr) }
        }
    }
}
```

The idea is to use the bulk of the implementation of that, generalize it to [`Vec`][Vec]s and [slices][slice], specialize it for [`&str`][str], provide documentation for both.

## [`Copy`][Copy] and [`Clone`][Clone]
[copy-clone]: #copy-clone

For the implementation of `From<&[T]> for Rc<[T]>`, `T` must be [`Copy`][Copy] if `ptr::copy_nonoverlapping` is used because this relies on it being memory safe to simply copy the bits over. If instead, [`T::clone()`][Clone] is used in a loop, then `T` can simply be [`Clone`][Clone] instead. This is however slower than using `ptr::copy_nonoverlapping`.

## [`Vec`][Vec] and [`Box`][Box]

For the implementation of `From<Vec<T>> for Rc<[T]>`, `T` need not be [`Copy`][Copy], nor [`Clone`][Clone]. The input vector already owns valid `T`s, and these elements are simply copied over bit for bit. After copying all elements, they are no longer
owned in the vector, which is then deallocated. Unfortunately, at this stage, the memory used by the vector can not be reused - this could potentially be changed in the future.

This is similar for [`Box`][Box].

## Suggested implementation

The actual implementations could / will look something like:

### For [`Rc`][Rc]

```rust
#[inline(always)]
unsafe fn slice_to_rc<'a, T, U, W, C>(src: &'a [T], cast: C, write_elems: W)
   -> Rc<U>
where U: ?Sized,
      W: FnOnce(&mut [T], &[T]),
      C: FnOnce(*mut RcBox<[T]>) -> *mut RcBox<U> {
    // Compute space to allocate for `RcBox<U>`.
    let susize = mem::size_of::<usize>();
    let aligned_len = 2 + (mem::size_of_val(src) + susize - 1) / susize;

    // Allocate enough space for `RcBox<U>`.
    let vec = RawVec::<usize>::with_capacity(aligned_len);
    let ptr = vec.ptr();
    forget(vec);

    // Combine the allocation address and the slice length into a
    // fat pointer to RcBox<[T]>.
    let rbp = slice::from_raw_parts_mut(ptr as *mut T, src.len())
                as *mut [T] as *mut RcBox<[T]>;

    // Initialize fields of RcBox<[T]>.
    (*rbp).strong.set(1);
    (*rbp).weak.set(1);
    write_elems(&mut (*rbp).value, src);

    // Recast to RcBox<U> and yield the Rc:
    let rcbox_ptr = cast(rbp);
    assert_eq!(aligned_len * susize, mem::size_of_val(&*rcbox_ptr));
    Rc { ptr: Shared::new(rcbox_ptr) }
}

#[unstable(feature = "shared_from_slice",
           reason = "TODO",
           issue = "TODO")]
impl<T> From<Vec<T>> for Rc<[T]> {
    /// Constructs a new `Rc<[T]>` from a `Vec<T>`.
    /// The allocated space of the `Vec<T>` is not reused,
    /// but new space is allocated and the old is deallocated.
    /// This happens due to the internal layout of `Rc`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(shared_from_slice)]
    /// use std::rc::Rc;
    ///
    /// let arr = [1, 2, 3];
    /// let vec = vec![Box::new(1), Box::new(2), Box::new(3)];
    /// let rc: Rc<[Box<usize>]> = Rc::from(vec);
    /// assert_eq!(rc.len(), arr.len());
    /// for (x, y) in rc.iter().zip(&arr) {
    ///     assert_eq!(**x, *y);
    /// }
    /// ```
    #[inline]
    fn from(mut vec: Vec<T>) -> Self {
        unsafe {
            let rc = slice_to_rc(vec.as_slice(), |p| p, |dst, src|
                ptr::copy_nonoverlapping(
                    src.as_ptr(), dst.as_mut_ptr(), src.len())
            );
            // Prevent vec from trying to drop the elements:
            vec.set_len(0);
            rc
        }
    }
}

#[unstable(feature = "shared_from_slice",
           reason = "TODO",
           issue = "TODO")]
impl<'a, T: Clone> From<&'a [T]> for Rc<[T]> {
    /// Constructs a new `Rc<[T]>` by cloning all elements from the shared slice
    /// [`&[T]`][slice]. The length of the reference counted slice will be exactly
    /// the given [slice].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(shared_from_slice)]
    /// use std::rc::Rc;
    ///
    /// #[derive(PartialEq, Clone, Debug)]
    /// struct Wrap(u8);
    ///
    /// let arr = [Wrap(1), Wrap(2), Wrap(3)];
    /// let rc: Rc<[Wrap]> = Rc::from(arr.as_ref());
    /// assert_eq!(rc.as_ref(), &arr);   // The elements match.
    /// assert_eq!(rc.len(), arr.len()); // The lengths match.
    /// ```
    ///
    /// Using the [`Into`][Into] trait:
    ///
    /// ```
    /// #![feature(shared_from_slice)]
    /// use std::rc::Rc;
    ///
    /// #[derive(PartialEq, Clone, Debug)]
    /// struct Wrap(u8);
    ///
    /// let rc: Rc<[Wrap]> = arr.as_ref().into();
    /// assert_eq!(rc.as_ref(), &arr);   // The elements match.
    /// assert_eq!(rc.len(), arr.len()); // The lengths match.
    /// ```
    ///
    /// [Into]: https://doc.rust-lang.org/std/convert/trait.Into.html
    /// [slice]: https://doc.rust-lang.org/std/primitive.slice.html
    #[inline]
    default fn from(slice: &'a [T]) -> Self {
        unsafe {
            slice_to_rc(slice, |p| p, |dst, src| {
                for (d, s) in dst.iter_mut().zip(src) {
                    ptr::write(d, s.clone())
                }
            })
        }
    }
}

#[unstable(feature = "shared_from_slice",
           reason = "TODO",
           issue = "TODO")]
impl<'a, T: Copy> From<&'a [T]> for Rc<[T]> {
    /// Constructs a new `Rc<[T]>` from a shared slice [`&[T]`][slice].
    /// All elements in the slice are copied and the length is exactly that of
    /// the given [slice]. In this case, `T` must be `Copy`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(shared_from_slice)]
    /// use std::rc::Rc;
    ///
    /// let arr = [1, 2, 3];
    /// let rc  = Rc::from(arr);
    /// assert_eq!(rc.as_ref(), &arr);   // The elements match.
    /// assert_eq!(rc.len(), arr.len()); // The length is the same.
    /// ```
    ///
    /// Using the [`Into`][Into] trait:
    ///
    /// ```
    /// #![feature(shared_from_slice)]
    /// use std::rc::Rc;
    ///
    /// let arr          = [1, 2, 3];
    /// let rc: Rc<[u8]> = arr.as_ref().into();
    /// assert_eq!(rc.as_ref(), &arr);   // The elements match.
    /// assert_eq!(rc.len(), arr.len()); // The length is the same.
    /// ```
    ///
    /// [Into]: ../../std/convert/trait.Into.html
    /// [slice]: ../../std/primitive.slice.html
    #[inline]
    fn from(slice: &'a [T]) -> Self {
        unsafe {
            slice_to_rc(slice, |p| p, <[T]>::copy_from_slice)
        }
    }
}

#[unstable(feature = "shared_from_slice",
           reason = "TODO",
           issue = "TODO")]
impl<'a> From<&'a str> for Rc<str> {
    /// Constructs a new `Rc<str>` from a [string slice].
    /// The underlying bytes are copied from it.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(shared_from_slice)]
    /// use std::rc::Rc;
    ///
    /// let slice = "hello world!";
    /// let rc: Rc<str> = Rc::from(slice);
    /// assert_eq!(rc.as_ref(), slice);    // The elements match.
    /// assert_eq!(rc.len(), slice.len()); // The length is the same.
    /// ```
    ///
    /// Using the [`Into`][Into] trait:
    ///
    /// ```
    /// #![feature(shared_from_slice)]
    /// use std::rc::Rc;
    ///
    /// let slice = "hello world!";
    /// let rc: Rc<str> = slice.into();
    /// assert_eq!(rc.as_ref(), slice);    // The elements match.
    /// assert_eq!(rc.len(), slice.len()); // The length is the same.
    /// ```
    ///
    /// This can be useful in doing [string interning], and caching your strings.
    ///
    /// ```
    /// // For Rc::ptr_eq
    /// #![feature(ptr_eq)]
    ///
    /// #![feature(shared_from_slice)]
    /// use std::rc::Rc;
    /// use std::collections::HashSet;
    /// use std::mem::drop;
    ///
    /// fn cache_str(cache: &mut HashSet<Rc<str>>, input: &str) -> Rc<str> {
    ///     // If the input hasn't been cached, do it:
    ///     if !cache.contains(input) {
    ///         cache.insert(input.into());
    ///     }
    ///
    ///     // Retrieve the cached element.
    ///     cache.get(input).unwrap().clone()
    /// }
    ///
    /// let first   = "hello world!";
    /// let second  = "goodbye!";
    /// let mut set = HashSet::new();
    ///
    /// // Cache the slices:
    /// let rc_first  = cache_str(&mut set, first);
    /// let rc_second = cache_str(&mut set, second);
    /// let rc_third  = cache_str(&mut set, second);
    ///
    /// // The contents match:
    /// assert_eq!(rc_first.as_ref(),  first);
    /// assert_eq!(rc_second.as_ref(), second);
    /// assert_eq!(rc_third.as_ref(),  rc_second.as_ref());
    ///
    /// // It was cached:
    /// assert_eq!(set.len(), 2);
    /// drop(set);
    /// assert_eq!(Rc::strong_count(&rc_first),  1);
    /// assert_eq!(Rc::strong_count(&rc_second), 2);
    /// assert_eq!(Rc::strong_count(&rc_third),  2);
    /// assert!(Rc::ptr_eq(&rc_second, &rc_third));
    ///
    /// [string interning]: https://en.wikipedia.org/wiki/String_interning
    fn from(slice: &'a str) -> Self {
        // This is safe since the input was valid utf8 to begin with, and thus
        // the invariants hold.
        unsafe {
            let bytes = slice.as_bytes();
            slice_to_rc(bytes, |p| p as *mut RcBox<str>, <[u8]>::copy_from_slice)
        }
    }
}

#[unstable(feature = "shared_from_slice",
           reason = "TODO",
           issue = "TODO")]
impl<T: ?Sized> From<Box<T>> for Rc<T> {
    /// Constructs a new `Rc<T>` from a `Box<T>` where `T` can be unsized.
    /// The allocated space of the `Box<T>` is not reused,
    /// but new space is allocated and the old is deallocated.
    /// This happens due to the internal layout of `Rc`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(shared_from_slice)]
    /// use std::rc::Rc;
    ///
    /// let arr = [1, 2, 3];
    /// let vec = vec![Box::new(1), Box::new(2), Box::new(3)].into_boxed_slice();
    /// let rc: Rc<[Box<usize>]> = Rc::from(vec);
    /// assert_eq!(rc.len(), arr.len());
    /// for (x, y) in rc.iter().zip(&arr) {
    ///     assert_eq!(**x, *y);
    /// }
    /// ```
    #[inline]
    fn from(boxed: Box<T>) -> Self {
        unsafe {
            // Compute space to allocate + alignment for `RcBox<T>`.
            let sizeb  = mem::size_of_val(&*boxed);
            let alignb = mem::align_of_val(&*boxed);
            let align  = cmp::max(alignb, mem::align_of::<usize>());
            let size   = offset_of_unsafe!(RcBox<T>, value) + sizeb;

            // Allocate the space.
            let alloc  = heap::allocate(size, align);

            // Cast to fat pointer: *mut RcBox<T>.
            let bptr      = Box::into_raw(boxed);
            let rcbox_ptr = {
                let mut tmp = bptr;
                ptr::write(&mut tmp as *mut _ as *mut * mut u8, alloc);
                tmp as *mut RcBox<T>
            };

            // Initialize fields of RcBox<T>.
            (*rcbox_ptr).strong.set(1);
            (*rcbox_ptr).weak.set(1);
            ptr::copy_nonoverlapping(
                bptr as *const u8,
                (&mut (*rcbox_ptr).value) as *mut T as *mut u8,
                sizeb);

            // Deallocate box, we've already forgotten it.
            heap::deallocate(bptr as *mut u8, sizeb, alignb);

            // Yield the Rc:
            assert_eq!(size, mem::size_of_val(&*rcbox_ptr));
            Rc { ptr: Shared::new(rcbox_ptr) }
        }
    }
}
```

These work on zero sized slices and vectors as well.

With more safe abstractions in the future, this can perhaps be rewritten with
less unsafe code. But this should not change the API itself and thus will never
cause a breaking change.

### For [`Arc`][Arc]

For the sake of brevity, just use the implementation above, and replace:
+ `slice_to_rc` with `slice_to_arc`,
+ `RcBox` with `ArcInner`,
+ `rcbox_ptr` with `arcinner_ptr`,
+ `Rc` with `Arc`.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

The documentation provided in the `impls` should be enough.

# Drawbacks
[drawbacks]: #drawbacks

The main drawback would be increasing the size of the standard library.

# Alternatives
[alternatives]: #alternatives

1. Only implement this for [`T: Copy`][Copy] and skip [`T: Clone`][Clone].
2. Let other libraries do this. This has the problems explained in the [motivation]
section above regarding [`RcBox`][RcBox] not being publically exposed as well as
the amount of feature gates needed to roll ones own [`Rc`][Rc] alternative - for
little gain.
3. Only implement this for [`Rc`][Rc] and skip it for [`Arc`][Arc].
4. Skip this for [`Vec`][Vec].
4. Only implement this for [`Vec`][Vec].
5. Skip this for [`Box`][Box].
6. Use [`AsRef`][AsRef]. For example: `impl<'a> From<&'a str> for Rc<str>` becomes `impl From<AsRef<str>> for Rc<str>`. It could potentially make the API a bit more ergonomic to use. However, it could run afoul of coherence issues, preventing other wanted impls. This RFC currently leans towards not using it.
7. Add these trait implementations of [`From`][From] as functions on [`&str`][str] like `.into_rc_str()` and on [`&[T]`][slice] like `.into_rc_slice()`.
This RFC currently leans towards using [`From`][From] implementations for the sake of uniformity and ergonomics. It also has the added benefit of letting you remember one method name instead of many. One could also consider [`String::into_boxed_str`][into_boxed_str] and [`Vec::into_boxed_slice`][into_boxed_slice], since these are similar with the difference being that this version uses the [`From`][From] trait, and is converted into a shared smart pointer instead.
8. **Also** add these APIs as [`associated functions`][associated functions] on [`Rc`][Rc] and [`Arc`][Arc] as follows:

```rust
impl<T: Clone> Rc<[T]> {
    fn from_slice(slice: &[T]) -> Self;
}

impl Rc<str> {
  fn from_str(slice: &str) -> Self;
}

impl<T: Clone> Arc<[T]> {
    fn from_slice(slice: &[T]) -> Self;
}

impl Arc<str> {
  fn from_str(slice: &str) -> Self;
}
```

# Unresolved questions
[unresolved]: #unresolved-questions

+ Should a special version of [`make_mut`][make_mut] be added for `Rc<[T]>`? This could look like:
```rust
impl<T> Rc<[T]> where T: Clone {
    fn make_mut_slice(this: &mut Rc<[T]>) -> &mut [T]
}
```

<!-- references -->
[Box]: https://doc.rust-lang.org/alloc/boxed/struct.Box.html
[Vec]: https://doc.rust-lang.org/std/collections/struct.HashSet.html
[Clone]: https://doc.rust-lang.org/std/clone/trait.Clone.html
[Copy]: https://doc.rust-lang.org/std/marker/trait.Copy.html
[From]: https://doc.rust-lang.org/std/convert/trait.From.html
[Rc]: https://doc.rust-lang.org/std/rc/struct.Rc.html
[Arc]: https://doc.rust-lang.org/std/sync/struct.Arc.html
[HashSet]: https://doc.rust-lang.org/std/collections/struct.HashSet.html
[str]: https://doc.rust-lang.org/std/primitive.str.html
[Path]: https://doc.rust-lang.org/std/path/struct.Path.html
[OsStr]: https://doc.rust-lang.org/std/ffi/struct.OsStr.html
[RcBox]: https://doc.rust-lang.org/src/alloc/rc.rs.html#242-246
[std::rc]: https://doc.rust-lang.org/std/rc/index.html
[slice]: https://doc.rust-lang.org/std/primitive.slice.html
[into_boxed_str]: https://doc.rust-lang.org/std/string/struct.String.html#method.into_boxed_str
[into_boxed_slice]: https://doc.rust-lang.org/std/vec/struct.Vec.html#method.into_boxed_slice
[AsRef]: https://doc.rust-lang.org/std/convert/trait.AsRef.html
[string interning]: https://en.wikipedia.org/wiki/String_interning
[tendril]: https://kmcallister.github.io/docs/html5ever/tendril/struct.Tendril.html
[Abstract Syntax Tree]: https://en.wikipedia.org/wiki/Abstract_syntax_tree
[XML]: https://en.wikipedia.org/wiki/XML
[namespace]: https://www.w3.org/TR/xml-names11/
[associated functions]: https://doc.rust-lang.org/book/method-syntax.html#associated-functions
[make_mut]: https://doc.rust-lang.org/stable/std/rc/struct.Rc.html#method.make_mut

<!-- references -->
