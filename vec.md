% Example: Implementing Vec

To bring everything together, we're going to write `std::Vec` from scratch.
Because all the best tools for writing unsafe code are unstable, this
project will only work on nightly (as of Rust 1.2.0).



# Layout

First off, we need to come up with the struct layout. Naively we want this
design:

```rust
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


```rust
#![feature(unique)]

use std::ptr::{Unique, self};

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

All of the `heap` API is totally unstable under the `heap_api` feature, though.
We could trivially define `heap::EMPTY` ourselves, but we'll want the rest of
the `heap` API anyway, so let's just get that dependency over with.




# Allocating Memory

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





# Push and Pop

Alright. We can initialize. We can allocate. Let's actually implement some
functionality! Let's start with `push`. All it needs to do is check if we're
full to grow, unconditionally write to the next index, and then increment our
length.

To do the write we have to be careful not to evaluate the memory we want to write
to. At worst, it's truly uninitialized memory from the allocator. At best it's the
bits of some old value we popped off. Either way, we can't just index to the memory
and dereference it, because that will evaluate the memory as a valid instance of
T. Worse, `foo[idx] = x` will try to call `drop` on the old value of `foo[idx]`!

The correct way to do this is with `ptr::write`, which just blindly overwrites the
target address with the bits of the value we provide. No evaluation involved.

For `push`, if the old len (before push was called) is 0, then we want to write
to the 0th index. So we should offset by the old len.

```rust
pub fn push(&mut self, elem: T) {
    if self.len == self.cap { self.grow(); }

    unsafe {
        ptr::write(self.ptr.offset(self.len as isize), elem);
    }

    // Can't fail, we'll OOM first.
    self.len += 1;
}
```

Easy! How about `pop`? Although this time the index we want to access is
initialized, Rust won't just let us dereference the location of memory to move
the value out, because that *would* leave the memory uninitialized! For this we
need `ptr::read`, which just copies out the bits from the target address and
intrprets it as a value of type T. This will leave the memory at this address
*logically* uninitialized, even though there is in fact a perfectly good instance
of T there.

For `pop`, if the old len is 1, we want to read out of the 0th index. So we
should offset by the *new* len.

```rust
pub fn pop(&mut self) -> Option<T> {
    if self.len == 0 {
        None
    } else {
        self.len -= 1;
        unsafe {
            Some(ptr::read(self.ptr.offset(self.len as isize)))
        }
    }
}
```





# Deallocating

Next we should implement Drop so that we don't massively leak tons of resources.
The easiest way is to just call `pop` until it yields None, and then deallocate
our buffer. Note that calling `pop` is uneeded if `T: !Drop`. In theory we can
ask Rust if T needs_drop and omit the calls to `pop`. However in practice LLVM
is *really* good at removing simple side-effect free code like this, so I wouldn't
bother unless you notice it's not being stripped (in this case it is).

We must not call `heap::deallocate` when `self.cap == 0`, as in this case we haven't
actually allocated any memory.


```rust
impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        if self.cap != 0 {
            while let Some(_) = self.pop() { }

            let align = mem::min_align_of::<T>();
            let elem_size = mem::size_of::<T>();
            let num_bytes = elem_size * self.cap;
            unsafe {
                heap::deallocate(*self.ptr, num_bytes, align);
            }
        }
    }
}
```





# Deref

Alright! We've got a decent minimal ArrayStack implemented. We can push, we can
pop, and we can clean up after ourselves. However there's a whole mess of functionality
we'd reasonably want. In particular, we have a proper array, but none of the slice
functionality. That's actually pretty easy to solve: we can implement `Deref<Target=[T]>`.
This will magically make our Vec coerce to and behave like a slice in all sorts of
conditions.

All we need is `slice::from_raw_parts`.

```rust
use std::ops::Deref;

impl<T> Deref for Vec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe {
            ::std::slice::from_raw_parts(*self.ptr, self.len)
        }
    }
}
```

And let's do DerefMut too:

```rust
use std::ops::DerefMut;

impl<T> DerefMut for Vec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            ::std::slice::from_raw_parts_mut(*self.ptr, self.len)
        }
    }
}
```

Now we have `len`, `first`, `last`, indexing, slicing, sorting, `iter`, `iter_mut`,
and all other sorts of bells and whistles provided by slice. Sweet!





# Insert and Remove

Something *not* provided but slice is `insert` and `remove`, so let's do those next.

Insert needs to shift all the elements at the target index to the right by one.
To do this we need to use `ptr::copy`, which is our version of C's `memmove`.
This copies some chunk of memory from one location to another, correctly handling
the case where the source and destination overlap (which will definitely happen
here).

If we insert at index `i`, we want to shift the `[i .. len]` to `[i+1 .. len+1]`
using the *old* len.

```rust
pub fn insert(&mut self, index: usize, elem: T) {
    // Note: `<=` because it's valid to insert after everything
    // which would be equivalent to push.
    assert!(index <= self.len, "index out of bounds");
    if self.cap == self.len { self.grow(); }

    unsafe {
        if index < self.len {
            // ptr::copy(src, dest, len): "copy from source to dest len elems"
            ptr::copy(self.ptr.offset(index as isize),
                      self.ptr.offset(index as isize + 1),
                      len - index);
        }
        ptr::write(self.ptr.offset(index as isize), elem);
        self.len += 1;
    }
}
```

Remove behaves in the opposite manner. We need to shift all the elements from
`[i+1 .. len + 1]` to `[i .. len]` using the *new* len.

```rust
pub fn remove(&mut self, index: usize) -> T {
    // Note: `<` because it's *not* valid to remove after everything
    assert!(index < self.len, "index out of bounds");
    unsafe {
        self.len -= 1;
        let result = ptr::read(self.ptr.offset(index as isize));
        ptr::copy(self.ptr.offset(index as isize + 1),
                  self.ptr.offset(index as isize),
                  len - index);
        result
    }
}
```





# IntoIter

Let's move on to writing iterators. `iter` and `iter_mut` have already been
written for us thanks to The Magic of Deref. However there's two interesting
iterators that Vec provides that slices can't: `into_iter` and `drain`.

IntoIter consumes the Vec by-value, and can consequently yield its elements
by-value. In order to enable this, IntoIter needs to take control of Vec's
allocation.

IntoIter needs to be DoubleEnded as well, to enable reading from both ends.
Reading from the back could just be implemented as calling `pop`, but reading
from the front is harder. We could call `remove(0)` but that would be insanely
expensive. Instead we're going to just use ptr::read to copy values out of either
end of the Vec without mutating the buffer at all.

To do this we're going to use a very common C idiom for array iteration. We'll
make two pointers; one that points to the start of the array, and one that points
to one-element past the end. When we want an element from one end, we'll read out
the value pointed to at that end and move the pointer over by one. When the two
pointers are equal, we know we're done.

Note that the order of read and offset are reversed for `next` and `next_back`
For `next_back` the pointer is always *after* the element it wants to read next,
while for `next` the pointer is always *at* the element it wants to read next.
To see why this is, consider the case where every element but one has been yielded.

The array looks like this:

```text
          S  E
[X, X, X, O, X, X, X]
```

If E pointed directly at the element it wanted to yield next, it would be
indistinguishable from the case where there are no more elements to yield.

So we're going to use the following struct:

```rust
struct IntoIter<T> {
    buf: Unique<T>,
    cap: usize,
    start: *const T,
    end: *const T,
}
```

One last subtle detail: if our Vec is empty, we want to produce an empty iterator.
This will actually technically fall out doing the naive thing of:

```text
start = ptr
end = ptr.offset(len)
```

However because `offset` is marked as a GEP inbounds instruction, this will tell
LLVM that ptr is allocated and won't alias other allocated memory. This is fine
for zero-sized types, as they can't alias anything. However if we're using
`heap::EMPTY` as a sentinel for a non-allocation for a *non-zero-sized* type,
this can cause undefined behaviour. Alas, we must therefore special case either
cap or len being 0 to not do the offset.

So this is what we end up with for initialization:

```rust
impl<T> Vec<T> {
    fn into_iter(self) -> IntoIter<T> {
        // Can't destructure Vec since it's Drop
        let ptr = self.ptr;
        let cap = self.cap;
        let len = self.len;

        // Make sure not to drop Vec since that will free the buffer
        mem::forget(self);

        unsafe {
            IntoIter {
                buf: ptr,
                cap: cap,
                start: *ptr,
                end: if cap == 0 {
                    // can't offset off this pointer, it's not allocated!
                    *ptr
                } else {
                    ptr.offset(len as isize)
                }
            }
        }
    }
}
```

Here's iterating forward:

```rust
impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                let result = ptr::read(self.start);
                self.start = self.start.offset(1);
                Some(result)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (self.end as usize - self.start as usize)
                  / mem::size_of::<T>();
        (len, Some(len))
    }
}
```

And here's iterating backwards.

```rust
impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<T> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                self.end = self.end.offset(-1);
                Some(ptr::read(self.end))
            }
        }
    }
}
```

Because IntoIter takes ownership of its allocation, it needs to implement Drop
to free it. However it *also* wants to implement Drop to drop any elements it
contains that weren't yielded.


```rust
impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        if self.cap != 0 {
            // drop any remaining elements
            for _ in &mut *self {}

            let align = mem::min_align_of::<T>();
            let elem_size = mem::size_of::<T>();
            let num_bytes = elem_size * self.cap;
            unsafe {
                heap::deallocate(*self.buf as *mut _, num_bytes, align);
            }
        }
    }
}
```

We've actually reached an interesting situation here: we've duplicated the logic
for specifying a buffer and freeing its memory. Now that we've implemented it and
identified *actual* logic duplication, this is a good time to perform some logic
compression.

We're going to abstract out the `(ptr, cap)` pair and give them the logic for
allocating, growing, and freeing:

```rust

struct RawVec<T> {
    ptr: Unique<T>,
    cap: usize,
}

impl<T> RawVec<T> {
    fn new() -> Self {
        assert!(mem::size_of::<T>() != 0, "TODO: implement ZST support");
        unsafe {
            RawVec { ptr: Unique::new(heap::EMPTY as *mut T), cap: 0 }
        }
    }

    // unchanged from Vec
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
}


impl<T> Drop for RawVec<T> {
    fn drop(&mut self) {
        if self.cap != 0 {
            let align = mem::min_align_of::<T>();
            let elem_size = mem::size_of::<T>();
            let num_bytes = elem_size * self.cap;
            unsafe {
                heap::deallocate(*self.ptr as *mut _, num_bytes, align);
            }
        }
    }
}
```

And change vec as follows:

```rust
pub struct Vec<T> {
    buf: RawVec<T>,
    len: usize,
}

impl<T> Vec<T> {
    fn ptr(&self) -> *mut T { *self.buf.ptr }

    fn cap(&self) -> usize { self.buf.cap }

    pub fn new() -> Self {
        Vec { buf: RawVec::new(), len: 0 }
    }

    // push/pop/insert/remove largely unchanged:
    // * `self.ptr -> self.ptr()`
    // * `self.cap -> self.cap()`
    // * `self.grow -> self.buf.grow()`
}

impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        while let Some(_) = self.pop() {}
        // deallocation is handled by RawVec
    }
}
```

And finally we can really simplify IntoIter:

```rust
struct IntoIter<T> {
    _buf: RawVec<T>, // we don't actually care about this. Just need it to live.
    start: *const T,
    end: *const T,
}

// next and next_back litterally unchanged since they never referred to the buf

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        // only need to ensure all our elements are read;
        // buffer will clean itself up afterwards.
        for _ in &mut *self {}
    }
}

impl<T> Vec<T> {
    pub fn into_iter(self) -> IntoIter<T> {
        unsafe {
            // need to use ptr::read to unsafely move the buf out since it's
            // not Copy.
            let buf = ptr::read(&self.buf);
            let len = self.len;
            mem::forget(self);

            IntoIter {
                start: *buf.ptr,
                end: buf.ptr.offset(len as isize),
                _buf: buf,
            }
        }
    }
}
```

Much better.





# Drain

Let's move on to Drain. Drain is largely the same as IntoIter, except that
instead of consuming the Vec, it borrows the Vec and leaves its allocation
free. For now we'll only implement the "basic" full-range version.

```rust,ignore
use std::marker::PhantomData;

struct Drain<'a, T: 'a> {
    vec: PhantomData<&'a mut Vec<T>>
    start: *const T,
    end: *const T,
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.start == self.end {
            None
```

-- wait, this is seeming familiar. Let's do some more compression. Both
IntoIter and Drain have the exact same structure, let's just factor it out.

```rust
struct RawValIter<T> {
    start: *const T,
    end: *const T,
}

impl<T> RawValIter<T> {
    // unsafe to construct because it has no associated lifetimes.
    // This is necessary to store a RawValIter in the same struct as
    // its actual allocation. OK since it's a private implementation
    // detail.
    unsafe fn new(slice: &[T]) -> Self {
        RawValIter {
            start: slice.as_ptr(),
            end: if slice.len() == 0 {
                slice.as_ptr()
            } else {
                slice.as_ptr().offset(slice.len() as isize)
            }
        }
    }
}

// Iterator and DoubleEndedIterator impls identical to IntoIter.
```

And IntoIter becomes the following:

```
pub struct IntoIter<T> {
    _buf: RawVec<T>, // we don't actually care about this. Just need it to live.
    iter: RawValIter<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<T> { self.iter.next_back() }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        for _ in &mut self.iter {}
    }
}

impl<T> Vec<T> {
    pub fn into_iter(self) -> IntoIter<T> {
        unsafe {
            let iter = RawValIter::new(&self);
            let buf = ptr::read(&self.buf);
            mem::forget(self);

            IntoIter {
                iter: iter,
                _buf: buf,
            }
        }
    }
}
```

Note that I've left a few quirks in this design to make upgrading Drain to work
with arbitrary subranges a bit easier. In particular we *could* have RawValIter
drain itself on drop, but that won't work right for a more complex Drain.
We also take a slice to simplify Drain initialization.

Alright, now Drain is really easy:

```rust
use std::marker::PhantomData;

pub struct Drain<'a, T: 'a> {
    vec: PhantomData<&'a mut Vec<T>>,
    iter: RawValIter<T>,
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> { self.iter.next_back() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    fn next_back(&mut self) -> Option<T> { self.iter.next_back() }
}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        for _ in &mut self.iter {}
    }
}

impl<T> Vec<T> {
    pub fn drain(&mut self) -> Drain<T> {
        // this is a mem::forget safety thing. If Drain is forgotten, we just
        // leak the whole Vec's contents. Also we need to do this *eventually*
        // anyway, so why not do it now?
        self.len = 0;

        unsafe {
            Drain {
                iter: RawValIter::new(&self),
                vec: PhantomData,
            }
        }
    }
}
```




# Handling Zero-Sized Types

It's time. We're going to fight the spectre that is zero-sized types. Safe Rust
*never* needs to care about this, but Vec is very intensive on raw pointers and
raw allocations, which are exactly the *only* two things that care about
zero-sized types. We need to be careful of two things:

* The raw allocator API has undefined behaviour if you pass in 0 for an
  allocation size.
* raw pointer offsets are no-ops for zero-sized types, which will break our
  C-style pointer iterator.

Thankfully we abstracted out pointer-iterators and allocating handling into
RawValIter and RawVec respectively. How mysteriously convenient.




## Allocating Zero-Sized Types

So if the allocator API doesn't support zero-sized allocations, what on earth
do we store as our allocation? Why, `heap::EMPTY` of course! Almost every operation
with a ZST is a no-op since ZSTs have exactly one value, and therefore no state needs
to be considered to store or load them. This actually extends to `ptr::read` and
`ptr::write`: they won't actually look at the pointer at all. As such we *never* need
to change the pointer.

Note however that our previous reliance on running out of memory before overflow is
no longer valid with zero-sized types. We must explicitly guard against capacity
overflow for zero-sized types.

Due to our current architecture, all this means is writing 3 guards, one in each
method of RawVec.

```rust
impl<T> RawVec<T> {
    fn new() -> Self {
        unsafe {
            // !0 is usize::MAX. This branch should be stripped at compile time.
            let cap = if mem::size_of::<T>() == 0 { !0 } else { 0 };

            // heap::EMPTY doubles as "unallocated" and "zero-sized allocation"
            RawVec { ptr: Unique::new(heap::EMPTY as *mut T), cap: cap }
        }
    }

    fn grow(&mut self) {
        unsafe {
            let elem_size = mem::size_of::<T>();

            // since we set the capacity to usize::MAX when elem_size is
            // 0, getting to here necessarily means the Vec is overfull.
            assert!(elem_size != 0, "capacity overflow");

            let align = mem::min_align_of::<T>();

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
}

impl<T> Drop for RawVec<T> {
    fn drop(&mut self) {
        let elem_size = mem::size_of::<T>();

        // don't free zero-sized allocations, as they were never allocated.
        if self.cap != 0 && elem_size != 0 {
            let align = mem::min_align_of::<T>();

            let num_bytes = elem_size * self.cap;
            unsafe {
                heap::deallocate(*self.ptr as *mut _, num_bytes, align);
            }
        }
    }
}
```

That's it. We support pushing and popping zero-sized types now. Our iterators
(that aren't provided by slice Deref) are still busted, though.




## Iterating Zero-Sized Types

Zero-sized offsets are no-ops. This means that our current design will always
initialize `start` and `end` as the same value, and our iterators will yield
nothing. The current solution to this is to cast the pointers to integers,
increment, and then cast them back:

```
impl<T> RawValIter<T> {
    unsafe fn new(slice: &[T]) -> Self {
        RawValIter {
            start: slice.as_ptr(),
            end: if mem::size_of::<T>() == 0 {
                ((slice.as_ptr() as usize) + slice.len()) as *const _
            } else if slice.len() == 0 {
                slice.as_ptr()
            } else {
                slice.as_ptr().offset(slice.len() as isize)
            }
        }
    }
}
```

Now we have a different bug. Instead of our iterators not running at all, our
iterators now run *forever*. We need to do the same trick in our iterator impls.
Also, our size_hint computation code will divide by 0 for ZSTs. Since we'll
basically be treating the two pointers as if they point to bytes, we'll just
map size 0 to divide by 1.

```
impl<T> Iterator for RawValIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                let result = ptr::read(self.start);
                self.start = if mem::size_of::<T>() == 0 {
                    (self.start as usize + 1) as *const _
                } else {
                    self.start.offset(1);
                }
                Some(result)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let elem_size = mem::size_of::<T>();
        let len = (self.end as usize - self.start as usize)
                  / if elem_size == 0 { 1 } else { elem_size };
        (len, Some(len))
    }
}

impl<T> DoubleEndedIterator for RawValIter<T> {
    fn next_back(&mut self) -> Option<T> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                self.end = if mem::size_of::<T>() == 0 {
                    (self.end as usize - 1) as *const _
                } else {
                    self.end.offset(-1);
                }
                Some(ptr::read(self.end))
            }
        }
    }
}
```

And that's it. Iteration works!



# Advanced Drain

TODO? Not clear if informative





# The Final Code

```rust
#![feature(unique)]
#![feature(heap_api)]

use std::ptr::{Unique, self};
use std::rt::heap;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;





struct RawVec<T> {
    ptr: Unique<T>,
    cap: usize,
}

impl<T> RawVec<T> {
    fn new() -> Self {
        unsafe {
            // !0 is usize::MAX. This branch should be stripped at compile time.
            let cap = if mem::size_of::<T>() == 0 { !0 } else { 0 };

            // heap::EMPTY doubles as "unallocated" and "zero-sized allocation"
            RawVec { ptr: Unique::new(heap::EMPTY as *mut T), cap: cap }
        }
    }

    fn grow(&mut self) {
        unsafe {
            let elem_size = mem::size_of::<T>();

            // since we set the capacity to usize::MAX when elem_size is
            // 0, getting to here necessarily means the Vec is overfull.
            assert!(elem_size != 0, "capacity overflow");

            let align = mem::min_align_of::<T>();

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
}

impl<T> Drop for RawVec<T> {
    fn drop(&mut self) {
        let elem_size = mem::size_of::<T>();
        if self.cap != 0 && elem_size != 0 {
            let align = mem::min_align_of::<T>();

            let num_bytes = elem_size * self.cap;
            unsafe {
                heap::deallocate(*self.ptr as *mut _, num_bytes, align);
            }
        }
    }
}





pub struct Vec<T> {
    buf: RawVec<T>,
    len: usize,
}

impl<T> Vec<T> {
    fn ptr(&self) -> *mut T { *self.buf.ptr }

    fn cap(&self) -> usize { self.buf.cap }

    pub fn new() -> Self {
        Vec { buf: RawVec::new(), len: 0 }
    }
    pub fn push(&mut self, elem: T) {
        if self.len == self.cap() { self.buf.grow(); }

        unsafe {
            ptr::write(self.ptr().offset(self.len as isize), elem);
        }

        // Can't fail, we'll OOM first.
        self.len += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe {
                Some(ptr::read(self.ptr().offset(self.len as isize)))
            }
        }
    }

    pub fn insert(&mut self, index: usize, elem: T) {
        assert!(index <= self.len, "index out of bounds");
        if self.cap() == self.len { self.buf.grow(); }

        unsafe {
            if index < self.len {
                ptr::copy(self.ptr().offset(index as isize),
                          self.ptr().offset(index as isize + 1),
                          self.len - index);
            }
            ptr::write(self.ptr().offset(index as isize), elem);
            self.len += 1;
        }
    }

    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len, "index out of bounds");
        unsafe {
            self.len -= 1;
            let result = ptr::read(self.ptr().offset(index as isize));
            ptr::copy(self.ptr().offset(index as isize + 1),
                      self.ptr().offset(index as isize),
                      self.len - index);
            result
        }
    }

    pub fn into_iter(self) -> IntoIter<T> {
        unsafe {
            let iter = RawValIter::new(&self);
            let buf = ptr::read(&self.buf);
            mem::forget(self);

            IntoIter {
                iter: iter,
                _buf: buf,
            }
        }
    }

    pub fn drain(&mut self) -> Drain<T> {
        // this is a mem::forget safety thing. If this is forgotten, we just
        // leak the whole Vec's contents. Also we need to do this *eventually*
        // anyway, so why not do it now?
        self.len = 0;
        unsafe {
            Drain {
                iter: RawValIter::new(&self),
                vec: PhantomData,
            }
        }
    }
}

impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        while let Some(_) = self.pop() {}
        // allocation is handled by RawVec
    }
}

impl<T> Deref for Vec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe {
            ::std::slice::from_raw_parts(self.ptr(), self.len)
        }
    }
}

impl<T> DerefMut for Vec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            ::std::slice::from_raw_parts_mut(self.ptr(), self.len)
        }
    }
}





struct RawValIter<T> {
    start: *const T,
    end: *const T,
}

impl<T> RawValIter<T> {
    unsafe fn new(slice: &[T]) -> Self {
        RawValIter {
            start: slice.as_ptr(),
            end: if mem::size_of::<T>() == 0 {
                ((slice.as_ptr() as usize) + slice.len()) as *const _
            } else if slice.len() == 0 {
                slice.as_ptr()
            } else {
                slice.as_ptr().offset(slice.len() as isize)
            }
        }
    }
}

impl<T> Iterator for RawValIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                let result = ptr::read(self.start);
                self.start = self.start.offset(1);
                Some(result)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let elem_size = mem::size_of::<T>();
        let len = (self.end as usize - self.start as usize)
                  / if elem_size == 0 { 1 } else { elem_size };
        (len, Some(len))
    }
}

impl<T> DoubleEndedIterator for RawValIter<T> {
    fn next_back(&mut self) -> Option<T> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                self.end = self.end.offset(-1);
                Some(ptr::read(self.end))
            }
        }
    }
}




pub struct IntoIter<T> {
    _buf: RawVec<T>, // we don't actually care about this. Just need it to live.
    iter: RawValIter<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<T> { self.iter.next_back() }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        for _ in &mut *self {}
    }
}




pub struct Drain<'a, T: 'a> {
    vec: PhantomData<&'a mut Vec<T>>,
    iter: RawValIter<T>,
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> { self.iter.next_back() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    fn next_back(&mut self) -> Option<T> { self.iter.next_back() }
}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        // pre-drain the iter
        for _ in &mut self.iter {}
    }
}

/// Abort the process, we're out of memory!
///
/// In practice this is probably dead code on most OSes
fn oom() {
    ::std::process::exit(-1);
}
```
