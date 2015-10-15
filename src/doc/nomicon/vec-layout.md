% Layout

First off, we need to come up with the struct layout. A Vec has three parts:
a pointer to the allocation, the size of the allocation, and the number of
elements that have been initialized.

Naively, this means we just want this design:

```rust
pub struct Vec<T> {
    ptr: *mut T,
    cap: usize,
    len: usize,
}
# fn main() {}
```

And indeed this would compile. Unfortunately, it would be incorrect. First, the
compiler will give us too strict variance. So a `&Vec<&'static str>`
couldn't be used where an `&Vec<&'a str>` was expected. More importantly, it
will give incorrect ownership information to the drop checker, as it will
conservatively assume we don't own any values of type `T`. See [the chapter
on ownership and lifetimes][ownership] for all the details on variance and
drop check.

As we saw in the ownership chapter, we should use `Unique<T>` in place of
`*mut T` when we have a raw pointer to an allocation we own. Unique is unstable,
so we'd like to not use it if possible, though.

As a recap, Unique is a wrapper around a raw pointer that declares that:

* We are variant over `T`
* We may own a value of type `T` (for drop check)
* We are Send/Sync if `T` is Send/Sync
* We deref to `*mut T` (so it largely acts like a `*mut` in our code)
* Our pointer is never null (so `Option<Vec<T>>` is null-pointer-optimized)

We can implement all of the above requirements except for the last
one in stable Rust:

```rust
use std::marker::PhantomData;
use std::ops::Deref;
use std::mem;

struct Unique<T> {
    ptr: *const T,              // *const for variance.
    _marker: PhantomData<T>,    // For the drop checker.
}

// Deriving Send and Sync is safe because we are the Unique owners
// of this data. It's like Unique<T> is "just" T.
unsafe impl<T: Send> Send for Unique<T> {}
unsafe impl<T: Sync> Sync for Unique<T> {}

impl<T> Unique<T> {
    pub fn new(ptr: *mut T) -> Self {
        Unique { ptr: ptr, _marker: PhantomData }
    }
}

impl<T> Deref for Unique<T> {
    type Target = *mut T;
    fn deref(&self) -> &*mut T {
        // There's no way to cast the *const to a *mut
        // while also taking a reference. So we just
        // transmute it since it's all "just pointers".
        unsafe { mem::transmute(&self.ptr) }
    }
}
# fn main() {}
```

Unfortunately the mechanism for stating that your value is non-zero is
unstable and unlikely to be stabilized soon. As such we're just going to
take the hit and use std's Unique:


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

If you don't care about the null-pointer optimization, then you can use the
stable code. However we will be designing the rest of the code around enabling
the optimization. In particular, `Unique::new` is unsafe to call, because
putting `null` inside of it is Undefined Behavior. Our stable Unique doesn't
need `new` to be unsafe because it doesn't make any interesting guarantees about
its contents.

[ownership]: ownership.html
