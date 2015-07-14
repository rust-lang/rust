% IntoIter

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

```rust,ignore
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

```rust,ignore
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

```rust,ignore
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

```rust,ignore
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


```rust,ignore
impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        if self.cap != 0 {
            // drop any remaining elements
            for _ in &mut *self {}

            let align = mem::align_of::<T>();
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

```rust,ignore

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
            let align = mem::align_of::<T>();
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
            let align = mem::align_of::<T>();
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

```rust,ignore
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

```rust,ignore
struct IntoIter<T> {
    _buf: RawVec<T>, // we don't actually care about this. Just need it to live.
    start: *const T,
    end: *const T,
}

// next and next_back literally unchanged since they never referred to the buf

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
