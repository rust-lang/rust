# `arbitrary_self_types_pointers`

The tracking issue for this feature is: [#44874]

[#38788]: https://github.com/rust-lang/rust/issues/44874

------------------------

This extends the [arbitrary self types] feature to allow methods to
receive `self` by pointer. For example:

```rust
#![feature(arbitrary_self_types_pointers)]

struct A;

impl A {
    fn m(self: *const Self) {}
}

fn main() {
    let a = A;
    let a_ptr: *const A = &a as *const A;
    a_ptr.m();
}
```

In general this is not advised: it's thought to be better practice to wrap
raw pointers in a newtype wrapper which implements the `core::ops::Receiver`
trait, then you need "only" the `arbitrary_self_types` feature. For example:

```rust
#![feature(arbitrary_self_types)]
#![allow(dead_code)]

struct A;

impl A {
    fn m(self: Wrapper<Self>) {} // can extract the pointer and do
        // what it needs
}

struct Wrapper<T>(*const T);

impl<T> core::ops::Receiver for Wrapper<T> {
    type Target = T;
}

fn main() {
    let a = A;
    let a_ptr: *const A = &a as *const A;
    let a_wrapper = Wrapper(a_ptr);
    a_wrapper.m();
}
```

[arbitrary self types]: arbitrary-self-types.md
