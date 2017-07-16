- Feature Name: alignto_intrinsic
- Start Date: 2017-06-20
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Add an intrinsic (`fn alignto(ptr: *const (), align: usize) -> usize`)
which returns the number of bytes that need to be skipped in order to correctly align the
pointer `ptr` to `align`.

Also add an `unsafe fn alignto<T, U>(&[U]) -> (&[U], &[T], &[U])` library function
under `core::mem` and `std::mem` that simplifies the common use case, returning
the unaligned prefix, the aligned center part and the unaligned trailing elements.
The function is unsafe because it essentially contains a `transmute<[U; N], T>`
if one reads from the aligned slice.

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
fn alignto(ptr: *const (), align: usize) -> usize;
```

which takes an arbitrary pointer it never reads from and a desired alignment
and returns the number of bytes that the pointer needs to be offset in order
to make it aligned to the desired alignment. It is perfectly valid for an
implementation to always yield `usize::max_value()` to signal that the pointer
cannot be aligned. Since the caller needs to check whether the returned offset
would be in-bounds of the allocation that the pointer points into, returning
`usize::max_value()` will never be in-bounds of the allocation and therefor
the caller cannot act upon the returned offset.

Most implementations will expand this intrinsic to

```rust
fn alignto(ptr: *const (), align: usize) -> usize {
    align - (ptr as usize % align)
}
```

## standard library functions

Add a new method `alignto` to `*const T` and `*mut T`, which forwards to the
`alignto`

Add two new functions `alignto` and `alignto_mut` to `core::mem` and `std::mem`
with the following signature:

```rust
unsafe fn alignto<T, U>(&[U]) -> (&[U], &[T], &[U]) { /**/ }
unsafe fn alignto_mut<T, U>(&mut [U]) -> (&mut [U], &mut [T], &mut [U]) { /**/ }
```

Calls to `alignto` are expanded to

```rust
unsafe fn alignto<T, U>(slice: &[U]) -> (&[U], &[T], &[U]) {
    if size_of::<T>() % size_of::<U>() == 0 {
        let align = core::mem::align_of::<T>();
        let size = core::mem::size_of::<T>();
        let source_size = core::mem::size_of::<U>();
        // number of bytes that need to be skipped until the pointer is aligned
        let offset = core::intrinsics::alignto(slice.as_ptr(), align);
        // if `align_of::<T>() <= align_of::<U>()`, or if pointer is accidentally aligned, then `offset == 0`
        //
        // due to `size_of::<T>() % size_of::<U>() == 0`,
        // the fact that `size_of::<U>() > align_of::<U>()`,
        // and the fact that `align_of::<T>() > align_of::<U>()` if `offset != 0` we know
        // that `offset % source_size == 0`
        let head_count = offset / source_size;
        let split_position = core::cmp::max(slice.len(), head_count);
        let (head, tail) = slice.split_at(split_position);
        // might be zero if not enough elements
        let mid_count = tail.len() * source_size / size;
        let mid = core::slice::from_raw_parts::<T>(tail.as_ptr() as *const _, mid_count);
        let tail = &tail[mid_count * core::mem::size_of::<T>()..];
        (head, mid, tail)
    } else {
        // can't properly fit a T into a sequence of `U`
        // FIXME: use GCD(size_of::<T>(), size_of::<U>()) as minimum `mid` size
        (slice, &[], &[])
    }
}
```

on all current platforms. `alignto_mut` is expanded accordingly.

Any backend may choose to always produce the following implementation, no matter
the given arguments, since the intrinsic is purely an optimization aid.

```rust
unsafe fn alignto<T, U>(slice: &[U]) -> (&[U], &[T], &[U]) {
    (slice, &[], &[])
}
```

Users of the functions must process all the returned slices and
cannot rely on any behaviour except that the `&[T]`'s elements are correctly
aligned and that all bytes of the original slice are present in the resulting
three slices.
The memory behind the reference must be treated as if created by
`transmute<[U; N], T>` with all the dangers that brings.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

## By example

On most platforms alignment is a well known concept independent of Rust.
Currently unsafe Rust code doing alignment checks needs to reproduce the known
patterns from C, which are hard to read and prone to errors when modified later.

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

One can clearly see the alignment check and the three slice parts.
The first part is the `else` branch of the outer `if align == 0`.
The (aligned) second part is the large `while` loop, and the third
art is the small `while` loop.

With the `alignto` method the code can be changed to

```rust
let ptr = v.as_ptr();
let align = ptr.offset(index).alignto(usize_bytes);
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

With the `alignto` function this could be written as


```rust
// Split `text` in three parts
// - unaligned initial part, before the first word aligned address in text
// - body, scan by 2 words at a time
// - the last remaining part, < 2 word size
let len = text.len();
let ptr = text.as_ptr();
let two_word_bytes = mem::size_of::<TwoWord>();

// choose alignment depending on platform
#[repr(align = 64)]
struct TwoWord([usize; 2]);

let (head, mid, tail) = std::mem::align_to<TwoWord>(text);

// search up to an aligned boundary
if let Some(index) = head.iter().position(|elt| *elt == x) {
    return Some(index);
}

// search the body of the text
let repeated_x = repeat_byte(x);

let position = mid.iter().position(|two| {
    // break if there is a matching byte
    let zu = contains_zero_byte(two.0[0] ^ repeated_x);
    let zv = contains_zero_byte(two.0[1] ^ repeated_x);
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

A lint could be implemented which detects hand-written alignment checks and
suggests to use the `alignto` function instead.

The `std::mem::align` function's documentation should point to `std::mem::alignto`
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
  is not part of a monomorphisation, since in that case `alignto` is statically
  known to never be effective
