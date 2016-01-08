% Insert and Remove

Something *not* provided by slice is `insert` and `remove`, so let's do those
next.

Insert needs to shift all the elements at the target index to the right by one.
To do this we need to use `ptr::copy`, which is our version of C's `memmove`.
This copies some chunk of memory from one location to another, correctly
handling the case where the source and destination overlap (which will
definitely happen here).

If we insert at index `i`, we want to shift the `[i .. len]` to `[i+1 .. len+1]`
using the old len.

```rust,ignore
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
                      self.len - index);
        }
        ptr::write(self.ptr.offset(index as isize), elem);
        self.len += 1;
    }
}
```

Remove behaves in the opposite manner. We need to shift all the elements from
`[i+1 .. len + 1]` to `[i .. len]` using the *new* len.

```rust,ignore
pub fn remove(&mut self, index: usize) -> T {
    // Note: `<` because it's *not* valid to remove after everything
    assert!(index < self.len, "index out of bounds");
    unsafe {
        self.len -= 1;
        let result = ptr::read(self.ptr.offset(index as isize));
        ptr::copy(self.ptr.offset(index as isize + 1),
                  self.ptr.offset(index as isize),
                  self.len - index);
        result
    }
}
```
