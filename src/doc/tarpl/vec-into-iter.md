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
expensive. Instead we're going to just use ptr::read to copy values out of
either end of the Vec without mutating the buffer at all.

To do this we're going to use a very common C idiom for array iteration. We'll
make two pointers; one that points to the start of the array, and one that
points to one-element past the end. When we want an element from one end, we'll
read out the value pointed to at that end and move the pointer over by one. When
the two pointers are equal, we know we're done.

Note that the order of read and offset are reversed for `next` and `next_back`
For `next_back` the pointer is always *after* the element it wants to read next,
while for `next` the pointer is always *at* the element it wants to read next.
To see why this is, consider the case where every element but one has been
yielded.

The array looks like this:

```text
          S  E
[X, X, X, O, X, X, X]
```

If E pointed directly at the element it wanted to yield next, it would be
indistinguishable from the case where there are no more elements to yield.

Although we don't actually care about it during iteration, we also need to hold
onto the Vec's allocation information in order to free it once IntoIter is
dropped.

So we're going to use the following struct:

```rust,ignore
struct IntoIter<T> {
    buf: Unique<T>,
    cap: usize,
    start: *const T,
    end: *const T,
}
```

One last subtle detail: if our Vec is empty, we want to produce an empty
iterator. This will actually technically fall out doing the naive thing of:

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
