- Feature Name: alignto_intrinsic
- Start Date: 2017-06-20
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Add an intrinsic (`fn alignto<T>(&[u8]) -> (&[u8], &[T], &[u8])`) which returns
a slice to correctly aligned objects of type `T` and the unalignable parts
before and after as `[u8]`. Also add an `unsafe` library function of the same
name and signature under `core::mem` and `std::mem`. The function is unsafe,
because it essentially contains a `transmute<[u8; N], T>` if one reads from
the aligned slice.

# Motivation
[motivation]: #motivation

The standard library (and most likely many crates) use code like

```rust
let is_aligned = (ptr as usize) & ((1 << (align - 1)) - 1) == 0;
let is_2_byte_aligned = ((ptr as usize + index) & (usize_bytes - 1)) == 0;
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

Add two new intrinsics `alignto` and `alignto_mut` with the following signature:

```rust
fn alignto<T>(&[u8]) -> (&[u8], &[T], &[u8]) { /**/ }
fn alignto_mut<T>(&mut [u8]) -> (&mut [u8], &mut [T], &mut [u8]) { /**/ }
```

Calls to `alignto` are expanded to

```rust
fn alignto<T>(slice: &[u8]) -> (&[u8], &[T], &[u8]) {
    let align = core::mem::align_of::<T>();
    let start = slice.as_ptr() as usize;
    let offset = start % align;
    let (head, tail) = if offset == 0 {
        (&[], slice)
    } else {
        slice.split_at(core::cmp::max(slice.len(), align - offset))
    };
    let count = tail.len() / core::mem::size_of::<T>();
    let mid = core::slice::from_raw_parts::<T>(tail.as_ptr() as *const _, count);
    let tail = &tail[count * core::mem::size_of::<T>()..];
    (head, mid, tail)
}
```

on all current platforms. `alignto_mut` is expanded accordingly.

A Rust backend may decide to expand it as

```rust
fn alignto<T>(slice: &[u8]) -> (&[u8], &[T], &[u8]) {
    (slice, &[], &[])
}
```

Users of the intrinsics and functions must process all the returned slices and
cannot rely on any behaviour except that the `&[T]`'s elements are correctly
aligned.

The intrinsics have `unsafe` functions wrappers in `core::mem` and `std::mem`
which simply forward to the intrinsics and which can be stabilized in the future.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

On most platforms alignment is a well known concept independent of Rust. The
important thing that has to be taught about these intrinsics is that they may not
ever actually yield an aligned slice. The functions and intrinsics may only be
used for optimizations.

A lint could be implemented which detects hand-written alignment checks and
suggests to use the `alignto` function instead. 

# Drawbacks
[drawbacks]: #drawbacks

None known to the author.

# Alternatives
[alternatives]: #alternatives

Miri could intercept calls to functions known to do alignment checks on pointers
and roll its own implementation for them. This doesn't scale well and is prone
to errors due to code duplication.

# Unresolved questions
[unresolved]: #unresolved-questions

None
