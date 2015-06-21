% Example: Implementing Vec

To bring everything together, we're going to write `std::Vec` from scratch.
Because the all the best tools for writing unsafe code are unstable, this
project will only work on nightly (as of Rust 1.2.0).

First off, we need to come up with the struct layout. Naively we want this
design:

```
struct Vec<T> {
	ptr: *mut T,
	cap: usize,
	len: usize,
}
```

And indeed this would compile. Unfortunately, it would be incorrect. The compiler
will give us too strict variance, so e.g. an `&Vec<&'static str>` couldn't be used
where an `&Vec<&'a str>` was expected. More importantly, it will give incorrect
ownership information to dropck, as it will conservatively assume we don't own
any values of type `T`. See [the chapter on ownership and lifetimes]
(lifetimes.html) for details.

As we saw in the lifetimes chapter, we should use `Unique<T>` in place of `*mut T`
when we have a raw pointer to an allocation we own:


```
#![feature(unique)]

use std::ptr::Unique;

pub struct Vec<T> {
    ptr: Unique<T>,
    cap: usize,
    len: usize,
}
```

As a recap, Unique is a wrapper around a raw pointer that declares that:

* We own at least one value of type `T`
* We are Send/Sync iff `T` is Send/Sync
* Our pointer is never null (and therefore `Option<Vec>` is null-pointer-optimized)

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

All of the `heap` API is totally unstable under the `alloc` feature, though.
We could trivially define `heap::EMPTY` ourselves, but we'll want the rest of
the `heap` API anyway, so let's just get that dependency over with.

So:

```rust
#![feature(alloc)]

use std::rt::heap::EMPTY;
use std::mem;

impl<T> Vec<T> {
	fn new() -> Self {
		assert!(mem::size_of::<T>() != 0, "We're not ready to handle ZSTs");
		unsafe {
			// need to cast EMPTY to the actual ptr type we want, let
			// inference handle it.
			Vec { ptr: Unique::new(heap::EMPTY as *mut _), len: 0, cap: 0 }
		}
	}
}
```

I slipped in that assert there because zero-sized types will require some
special handling throughout our code, and I want to defer the issue for now.
Without this assert, some of our early drafts will do some Very Bad Things.

Next we need to figure out what to actually do when we *do* want space. For that,
we'll need to use the rest of the heap APIs. These basically allow us to
talk directly to Rust's instance of jemalloc.

We'll also need a way to handle out-of-memory conditions. The standard library
calls the `abort` intrinsic, but calling intrinsics from normal Rust code is a
pretty bad idea. Unfortunately, the `abort` exposed by the standard library
allocates. Not something we want to do during `oom`! Instead, we'll call
`std::process::exit`.

```rust
fn oom() {
    ::std::process::exit(-9999);
}
```

Okay, now we can write growing:

```rust
fn grow(&mut self) {
    unsafe {
        let align = mem::min_align_of::<T>();
        let elem_size = mem::size_of::<T>();

        let (new_cap, ptr) = if self.cap == 0 {
            let ptr = heap::allocate(elem_size, align);
            (1, ptr)
        } else {
            let new_cap = 2 * self.cap;
            let ptr = heap::reallocate(*self.ptr as *mut _,
                                        self.cap * elem_size,
                                        new_cap * elem_size,
                                        align);
            (new_cap, ptr)
        };

        // If allocate or reallocate fail, we'll get `null` back
        if ptr.is_null() { oom() }

        self.ptr = Unique::new(ptr as *mut _);
        self.cap = new_cap;
    }
}
```

There's nothing particularly tricky in here: if we're totally empty, we need
to do a fresh allocation. Otherwise, we need to reallocate the current pointer.
Although we have a subtle bug here with the multiply overflow.

TODO: rest of this


