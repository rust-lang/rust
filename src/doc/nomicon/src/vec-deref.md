# Deref

Alright! We've got a decent minimal stack implemented. We can push, we can
pop, and we can clean up after ourselves. However there's a whole mess of
functionality we'd reasonably want. In particular, we have a proper array, but
none of the slice functionality. That's actually pretty easy to solve: we can
implement `Deref<Target=[T]>`. This will magically make our Vec coerce to, and
behave like, a slice in all sorts of conditions.

All we need is `slice::from_raw_parts`. It will correctly handle empty slices
for us. Later once we set up zero-sized type support it will also Just Work
for those too.

```rust,ignore
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

```rust,ignore
use std::ops::DerefMut;

impl<T> DerefMut for Vec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            ::std::slice::from_raw_parts_mut(*self.ptr, self.len)
        }
    }
}
```

Now we have `len`, `first`, `last`, indexing, slicing, sorting, `iter`,
`iter_mut`, and all other sorts of bells and whistles provided by slice. Sweet!
