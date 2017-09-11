- Feature Name: align_to_intrinsic
- Start Date: 2017-06-20
- RFC PR: https://github.com/rust-lang/rfcs/pull/2043
- Rust Issue: https://github.com/rust-lang/rust/issues/44488

# Summary
[summary]: #summary

Add an intrinsic (`fn align_offset(ptr: *const (), align: usize) -> usize`)
which returns the number of bytes that need to be skipped in order to correctly align the
pointer `ptr` to `align`.

The intrinsic is reexported as a method on `*const T` and `*mut T`.

Also add an `unsafe fn align_to<U>(&self) -> (&[T], &[U], &[T])` method to `[T]`.
The method simplifies the common use case, returning
the unaligned prefix, the aligned center part and the unaligned trailing elements.
The function is unsafe because it produces a `&U` to the memory location of a `T`,
which might expose padding bytes or violate invariants of `T` or `U`.

# Motivation
[motivation]: #motivation

The standard library (and most likely many crates) use code like

```rust
let is_aligned = (ptr as usize) & ((1 << (align - 1)) - 1) == 0;
let is_2_word_aligned = ((ptr as usize + index) & (usize_bytes - 1)) == 0;
let is_t_aligned = ((ptr as usize) % std::mem::align_of::<T>()) == 0;
```

to check whether a pointer is aligned in order to perform optimizations like
reading multiple bytes at once. Not only is this code which is easy to get
wrong, and which is hard to read (and thus increasing the chance of future breakage)
but it also makes it impossible for `miri` to evaluate such statements. This
means that `miri` cannot do utf8-checking, since that code contains such
optimizations. Without utf8-checking, Rustc's future const evaluation would not
be able to convert a `[u8]` into a `str`.

# Detailed design
[design]: #detailed-design

## supporting intrinsic

Add a new intrinsic

```rust
fn align_offset(ptr: *const (), align: usize) -> usize;
```

which takes an arbitrary pointer it never reads from and a desired alignment
and returns the number of bytes that the pointer needs to be offset in order
to make it aligned to the desired alignment. It is perfectly valid for an
implementation to always yield `usize::max_value()` to signal that the pointer
cannot be aligned. Since the caller needs to check whether the returned offset
would be in-bounds of the allocation that the pointer points into, returning
`usize::max_value()` will never be in-bounds of the allocation and therefor
the caller cannot act upon the returned offset.

It might be expected that the maximum offset returned is `align - 1`, but as
the motivation of the rfc states, `miri` cannot guarantee that a pointer can
be aligned irrelevant of the operations done on it.

Most implementations will expand this intrinsic to

```rust
fn align_offset(ptr: *const (), align: usize) -> usize {
    let offset = ptr as usize % align;
    if offset == 0 {
        0
    } else {
        align - offset
    }
}
```

The `align` parameter must be a power of two and smaller than `2^32`.
Usually one should pass in the result of an `align_of` call.

## standard library functions

Add a new method `align_offset` to `*const T` and `*mut T`, which forwards to the
`align_offset` intrinsic.

Add two new methods `align_to` and `align_to_mut` to the slice type.

```rust
impl<T> [T] {
    /* ... other methods ... */
    unsafe fn align_to<U>(&self) -> (&[T], &[U], &[T]) { /**/ }
    unsafe fn align_to_mut<U>(&mut self) -> (&mut [T], &mut [U], &mut [T]) { /**/ }
}
```

`align_to` can be implemented as

```rust
unsafe fn align_to<U>(&self) -> (&[T], &[U], &[T]) {
    use core::mem::{size_of, align_of};
    assert!(size_of::<U>() != 0 && size_of::<T>() != 0, "don't use `align_to` with zsts");
    if size_of::<U>() % size_of::<T>() == 0 {
        let align = align_of::<U>();
        let size = size_of::<U>();
        let source_size = size_of::<T>();
        // number of bytes that need to be skipped until the pointer is aligned
        let offset = self.as_ptr().align_offset(align);
        // if `align_of::<U>() <= align_of::<T>()`, or if pointer is accidentally aligned, then `offset == 0`
        //
        // due to `size_of::<U>() % size_of::<T>() == 0`,
        // the fact that `size_of::<T>() > align_of::<T>()`,
        // and the fact that `align_of::<U>() > align_of::<T>()` if `offset != 0` we know
        // that `offset % source_size == 0`
        let head_count = offset / source_size;
        let split_position = core::cmp::max(self.len(), head_count);
        let (head, tail) = self.split_at(split_position);
        // might be zero if not enough elements
        let mid_count = tail.len() * source_size / size;
        let mid = core::slice::from_raw_parts::<U>(tail.as_ptr() as *const _, mid_count);
        let tail = &tail[mid_count * size_of::<U>()..];
        (head, mid, tail)
    } else {
        // can't properly fit a U into a sequence of `T`
        // FIXME: use GCD(size_of::<U>(), size_of::<T>()) as minimum `mid` size
        (self, &[], &[])
    }
}
```

on all current platforms. `align_to_mut` is expanded accordingly.

Users of the functions must process all the returned slices and
cannot rely on any behaviour except that the `&[U]`'s elements are correctly
aligned and that all bytes of the original slice are present in the resulting
three slices.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

## By example

On most platforms alignment is a well known concept independent of Rust.
Currently unsafe Rust code doing alignment checks needs to reproduce the known
patterns from C, which are hard to read and prone to errors when modified later.

Thus, whenever pointers need to be manually aligned, the developer is given a
choice:

1. In the case where processing the initial unaligned bits might abort the entire
   process, use `align_offset`
2. If it is likely that all bytes are going to get processed, use `align_to`
    * `align_to` has a slight overhead for creating the slices in case not all
        slices are used

### Example 1 (pointers)

The standard library uses an alignment optimization for quickly
skipping over ascii code during utf8 checking a byte slice. The current code
looks as follows:

```rust
// Ascii case, try to skip forward quickly.
// When the pointer is aligned, read 2 words of data per iteration
// until we find a word containing a non-ascii byte.
let ptr = v.as_ptr();
let align = (ptr as usize + index) & (usize_bytes - 1);

```

With the `align_offset` method the code can be changed to

```rust
let ptr = v.as_ptr();
let align = unsafe {
    // the offset is safe, because `index` is guaranteed inbounds
    ptr.offset(index).align_offset(usize_bytes)
};
```

## Example 2 (slices)

The `memchr` impl in the standard library explicitly uses the three phases of
the `align_to` functions:

```rust
// Split `text` in three parts
// - unaligned initial part, before the first word aligned address in text
// - body, scan by 2 words at a time
// - the last remaining part, < 2 word size
let len = text.len();
let ptr = text.as_ptr();
let usize_bytes = mem::size_of::<usize>();

// search up to an aligned boundary
let align = (ptr as usize) & (usize_bytes- 1);
let mut offset;
if align > 0 {
    offset = cmp::min(usize_bytes - align, len);
    if let Some(index) = text[..offset].iter().position(|elt| *elt == x) {
        return Some(index);
    }
} else {
    offset = 0;
}

// search the body of the text
let repeated_x = repeat_byte(x);

if len >= 2 * usize_bytes {
    while offset <= len - 2 * usize_bytes {
        unsafe {
            let u = *(ptr.offset(offset as isize) as *const usize);
            let v = *(ptr.offset((offset + usize_bytes) as isize) as *const usize);

            // break if there is a matching byte
            let zu = contains_zero_byte(u ^ repeated_x);
            let zv = contains_zero_byte(v ^ repeated_x);
            if zu || zv {
                break;
            }
        }
        offset += usize_bytes * 2;
    }
}

// find the byte after the point the body loop stopped
text[offset..].iter().position(|elt| *elt == x).map(|i| offset + i)
```

With the `align_to` function this could be written as


```rust
// Split `text` in three parts
// - unaligned initial part, before the first word aligned address in text
// - body, scan by 2 words at a time
// - the last remaining part, < 2 word size
let len = text.len();
let ptr = text.as_ptr();

let (head, mid, tail) = text.align_to::<(usize, usize)>();

// search up to an aligned boundary
if let Some(index) = head.iter().position(|elt| *elt == x) {
    return Some(index);
}

// search the body of the text
let repeated_x = repeat_byte(x);

let position = mid.iter().position(|two| {
    // break if there is a matching byte
    let zu = contains_zero_byte(two.0 ^ repeated_x);
    let zv = contains_zero_byte(two.1 ^ repeated_x);
    zu || zv
});

if let Some(index) = position {
    let offset = index * two_word_bytes + head.len();
    return text[offset..].iter().position(|elt| *elt == x).map(|i| offset + i)
}

// find the byte in the trailing unaligned part
tail.iter().position(|elt| *elt == x).map(|i| head.len() + mid.len() + i)
```

## Documentation

A lint could be added to `clippy` which detects hand-written alignment checks and
suggests to use the `align_to` function instead.

The `std::mem::align` function's documentation should point to `[T]::align_to`
in order to increase the visibility of the function. The documentation of
`std::mem::align` should note that it is unidiomatic to manually align pointers,
since that might not be supported on all platforms and is prone to implementation
errors.

# Drawbacks
[drawbacks]: #drawbacks

None known to the author.

# Alternatives
[alternatives]: #alternatives

## Duplicate functions without optimizations for miri

Miri could intercept calls to functions known to do alignment checks on pointers
and roll its own implementation for them. This doesn't scale well and is prone
to errors due to code duplication.

# Unresolved questions
[unresolved]: #unresolved-questions

* produce a lint in case `sizeof<T>() % sizeof<U>() != 0` and in case the expansion
  is not part of a monomorphisation, since in that case `align_to` is statically
  known to never be effective
