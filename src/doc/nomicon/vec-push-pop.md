% Push and Pop

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

```rust,ignore
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
the value out, because that would leave the memory uninitialized! For this we
need `ptr::read`, which just copies out the bits from the target address and
interprets it as a value of type T. This will leave the memory at this address
logically uninitialized, even though there is in fact a perfectly good instance
of T there.

For `pop`, if the old len is 1, we want to read out of the 0th index. So we
should offset by the new len.

```rust,ignore
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
