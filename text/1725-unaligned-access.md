- Feature Name: `unaligned_access`
- Start Date: 2016-08-22
- RFC PR: [rust-lang/rfcs#1725](https://github.com/rust-lang/rfcs/pull/1725)
- Rust Issue: [rust-lang/rust#37955](https://github.com/rust-lang/rust/issues/37955)

# Summary
[summary]: #summary

Add two functions, `ptr::read_unaligned` and `ptr::write_unaligned`, which allows reading/writing to an unaligned pointer. All other functions that access memory (`ptr::{read,write}`, `ptr::copy{_nonoverlapping}`, etc) require that a pointer be suitably aligned for its type.

# Motivation
[motivation]: #motivation

One major use case is to make working with packed structs easier:

```rust
#[repr(packed)]
struct Packed(u8, u16, u8);

let mut a = Packed(0, 1, 0);
unsafe {
    let b = ptr::read_unaligned(&a.1);
    ptr::write_unaligned(&mut a.1, b + 1);
}
```

Other use cases generally involve parsing some file formats or network protocols that use unaligned values.

# Detailed design
[design]: #detailed-design

The implementation of these functions are simple wrappers around `ptr::copy_nonoverlapping`. The pointers are cast to `u8` to ensure that LLVM does not make any assumptions about the alignment.

```rust
pub unsafe fn read_unaligned<T>(p: *const T) -> T {
    let mut r = mem::uninitialized();
    ptr::copy_nonoverlapping(p as *const u8,
                             &mut r as *mut _ as *mut u8,
                             mem::size_of::<T>());
    r
}

pub unsafe fn write_unaligned<T>(p: *mut T, v: T) {
    ptr::copy_nonoverlapping(&v as *const _ as *const u8,
                             p as *mut u8,
                             mem::size_of::<T>());
}
```

# Drawbacks
[drawbacks]: #drawbacks

There functions aren't *stricly* necessary since they are just convenience wrappers around `ptr::copy_nonoverlapping`.

# Alternatives
[alternatives]: #alternatives

We could simply not add these, however figuring out how to do unaligned access properly is extremely unintuitive: you need to cast the pointer to `*mut u8` and then call `ptr::copy_nonoverlapping`.

# Unresolved questions
[unresolved]: #unresolved-questions

None
