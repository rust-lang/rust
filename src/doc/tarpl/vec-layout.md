% Layout

First off, we need to come up with the struct layout. Naively we want this
design:

```rust
pub struct Vec<T> {
    ptr: *mut T,
    cap: usize,
    len: usize,
}

# fn main() {}
```

And indeed this would compile. Unfortunately, it would be incorrect. The
compiler will give us too strict variance, so e.g. an `&Vec<&'static str>`
couldn't be used where an `&Vec<&'a str>` was expected. More importantly, it
will give incorrect ownership information to dropck, as it will conservatively
assume we don't own any values of type `T`. See [the chapter on ownership and
lifetimes] (lifetimes.html) for details.

As we saw in the lifetimes chapter, we should use `Unique<T>` in place of
`*mut T` when we have a raw pointer to an allocation we own:


```rust
#![feature(unique)]

use std::ptr::{Unique, self};

pub struct Vec<T> {
    ptr: Unique<T>,
    cap: usize,
    len: usize,
}

# fn main() {}
```

As a recap, Unique is a wrapper around a raw pointer that declares that:

* We may own a value of type `T`
* We are Send/Sync iff `T` is Send/Sync
* Our pointer is never null (and therefore `Option<Vec>` is
  null-pointer-optimized)

That last point is subtle. First, it makes `Unique::new` unsafe to call, because
putting `null` inside of it is Undefined Behaviour. It also throws a
wrench in an important feature of Vec (and indeed all of the std collections):
an empty Vec doesn't actually allocate at all. So if we can't allocate,
but also can't put a null pointer in `ptr`, what do we do in
`Vec::new`? Well, we just put some other garbage in there!

This is perfectly fine because we already have `cap == 0` as our sentinel for no
allocation. We don't even need to handle it specially in almost any code because
we usually need to check if `cap > len` or `len > 0` anyway. The traditional
Rust value to put here is `0x01`. The standard library actually exposes this
as `std::rt::heap::EMPTY`. There are quite a few places where we'll want to use
`heap::EMPTY` because there's no real allocation to talk about but `null` would
make the compiler angry.

All of the `heap` API is totally unstable under the `heap_api` feature, though.
We could trivially define `heap::EMPTY` ourselves, but we'll want the rest of
the `heap` API anyway, so let's just get that dependency over with.

