% Allocating Memory

So:

```rust
#![feature(heap_api)]

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

Okay, now we can write growing. Roughly, we want to have this logic:

```text
if cap == 0:
    allocate()
    cap = 1
else
    reallocate
    cap *= 2
```

But Rust's only supported allocator API is so low level that we'll need to
do a fair bit of extra work, though. We also need to guard against some special
conditions that can occur with really large allocations. In particular, we index
into arrays using unsigned integers, but `ptr::offset` takes signed integers. This
means Bad Things will happen if we ever manage to grow to contain more than
`isize::MAX` elements. Thankfully, this isn't something we need to worry about
in most cases.

On 64-bit targets we're artifically limited to only 48-bits, so we'll run out
of memory far before we reach that point. However on 32-bit targets, particularly
those with extensions to use more of the address space, it's theoretically possible
to successfully allocate more than `isize::MAX` bytes of memory. Still, we only
really need to worry about that if we're allocating elements that are a byte large.
Anything else will use up too much space.

However since this is a tutorial, we're not going to be particularly optimal here,
and just unconditionally check, rather than use clever platform-specific `cfg`s.

```rust
fn grow(&mut self) {
    // this is all pretty delicate, so let's say it's all unsafe
    unsafe {
        let align = mem::min_align_of::<T>();
        let elem_size = mem::size_of::<T>();

        let (new_cap, ptr) = if self.cap == 0 {
            let ptr = heap::allocate(elem_size, align);
            (1, ptr)
        } else {
            // as an invariant, we can assume that `self.cap < isize::MAX`,
            // so this doesn't need to be checked.
            let new_cap = self.cap * 2;
            // Similarly this can't overflow due to previously allocating this
            let old_num_bytes = self.cap * elem_size;

            // check that the new allocation doesn't exceed `isize::MAX` at all
            // regardless of the actual size of the capacity. This combines the
            // `new_cap <= isize::MAX` and `new_num_bytes <= usize::MAX` checks
            // we need to make. We lose the ability to allocate e.g. 2/3rds of
            // the address space with a single Vec of i16's on 32-bit though.
            // Alas, poor Yorick -- I knew him, Horatio.
            assert!(old_num_bytes <= (::std::isize::MAX as usize) / 2,
                    "capacity overflow");

            let new_num_bytes = old_num_bytes * 2;
            let ptr = heap::reallocate(*self.ptr as *mut _,
                                        old_num_bytes,
                                        new_num_bytes,
                                        align);
            (new_cap, ptr)
        };

        // If allocate or reallocate fail, we'll get `null` back
        if ptr.is_null() { oom(); }

        self.ptr = Unique::new(ptr as *mut _);
        self.cap = new_cap;
    }
}
```

Nothing particularly tricky here. Just computing sizes and alignments and doing
some careful multiplication checks.

