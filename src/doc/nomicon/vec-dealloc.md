% Deallocating

Next we should implement Drop so that we don't massively leak tons of resources.
The easiest way is to just call `pop` until it yields None, and then deallocate
our buffer. Note that calling `pop` is unneeded if `T: !Drop`. In theory we can
ask Rust if `T` `needs_drop` and omit the calls to `pop`. However in practice
LLVM is *really* good at removing simple side-effect free code like this, so I
wouldn't bother unless you notice it's not being stripped (in this case it is).

We must not call `heap::deallocate` when `self.cap == 0`, as in this case we
haven't actually allocated any memory.


```rust,ignore
impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        if self.cap != 0 {
            while let Some(_) = self.pop() { }

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
