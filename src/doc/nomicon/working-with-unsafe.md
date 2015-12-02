% Working with Unsafe

Rust generally only gives us the tools to talk about Unsafe Rust in a scoped and
binary manner. Unfortunately, reality is significantly more complicated than
that. For instance, consider the following toy function:

```rust
fn index(idx: usize, arr: &[u8]) -> Option<u8> {
    if idx < arr.len() {
        unsafe {
            Some(*arr.get_unchecked(idx))
        }
    } else {
        None
    }
}
```

Clearly, this function is safe. We check that the index is in bounds, and if it
is, index into the array in an unchecked manner. But even in such a trivial
function, the scope of the unsafe block is questionable. Consider changing the
`<` to a `<=`:

```rust
fn index(idx: usize, arr: &[u8]) -> Option<u8> {
    if idx <= arr.len() {
        unsafe {
            Some(*arr.get_unchecked(idx))
        }
    } else {
        None
    }
}
```

This program is now unsound, and yet *we only modified safe code*. This is the
fundamental problem of safety: it's non-local. The soundness of our unsafe
operations necessarily depends on the state established by otherwise
"safe" operations.

Safety is modular in the sense that opting into unsafety doesn't require you
to consider arbitrary other kinds of badness. For instance, doing an unchecked
index into a slice doesn't mean you suddenly need to worry about the slice being
null or containing uninitialized memory. Nothing fundamentally changes. However
safety *isn't* modular in the sense that programs are inherently stateful and
your unsafe operations may depend on arbitrary other state.

Trickier than that is when we get into actual statefulness. Consider a simple
implementation of `Vec`:

```rust
use std::ptr;

// Note this definition is insufficient. See the section on implementing Vec.
pub struct Vec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

// Note this implementation does not correctly handle zero-sized types.
// We currently live in a nice imaginary world of only positive fixed-size
// types.
impl<T> Vec<T> {
    pub fn push(&mut self, elem: T) {
        if self.len == self.cap {
            // Not important for this example.
            self.reallocate();
        }
        unsafe {
            ptr::write(self.ptr.offset(self.len as isize), elem);
            self.len += 1;
        }
    }

    # fn reallocate(&mut self) { }
}

# fn main() {}
```

This code is simple enough to reasonably audit and verify. Now consider
adding the following method:

```rust,ignore
fn make_room(&mut self) {
    // Grow the capacity.
    self.cap += 1;
}
```

This code is 100% Safe Rust but it is also completely unsound. Changing the
capacity violates the invariants of Vec (that `cap` reflects the allocated space
in the Vec). This is not something the rest of Vec can guard against. It *has*
to trust the capacity field because there's no way to verify it.

`unsafe` does more than pollute a whole function: it pollutes a whole *module*.
Generally, the only bullet-proof way to limit the scope of unsafe code is at the
module boundary with privacy.

However this works *perfectly*. The existence of `make_room` is *not* a
problem for the soundness of Vec because we didn't mark it as public. Only the
module that defines this function can call it. Also, `make_room` directly
accesses the private fields of Vec, so it can only be written in the same module
as Vec.

It is therefore possible for us to write a completely safe abstraction that
relies on complex invariants. This is *critical* to the relationship between
Safe Rust and Unsafe Rust. We have already seen that Unsafe code must trust
*some* Safe code, but can't trust *generic* Safe code. It can't trust an
arbitrary implementor of a trait or any function that was passed to it to be
well-behaved in a way that safe code doesn't care about.

However if unsafe code couldn't prevent client safe code from messing with its
state in arbitrary ways, safety would be a lost cause. Thankfully, it *can*
prevent arbitrary code from messing with critical state due to privacy.

Safety lives!

