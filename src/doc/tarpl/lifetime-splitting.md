% Splitting Lifetimes

The mutual exclusion property of mutable references can be very limiting when
working with a composite structure. The borrow checker understands some basic stuff, but
will fall over pretty easily. It *does* understand structs sufficiently to
know that it's possible to borrow disjoint fields of a struct simultaneously.
So this works today:

```rust
struct Foo {
    a: i32,
    b: i32,
    c: i32,
}

let mut x = Foo {a: 0, b: 0, c: 0};
let a = &mut x.a;
let b = &mut x.b;
let c = &x.c;
*b += 1;
let c2 = &x.c;
*a += 10;
println!("{} {} {} {}", a, b, c, c2);
```

However borrowck doesn't understand arrays or slices in any way, so this doesn't
work:

```rust,ignore
let x = [1, 2, 3];
let a = &mut x[0];
let b = &mut x[1];
println!("{} {}", a, b);
```

```text
<anon>:3:18: 3:22 error: cannot borrow immutable indexed content `x[..]` as mutable
<anon>:3     let a = &mut x[0];
                          ^~~~
<anon>:4:18: 4:22 error: cannot borrow immutable indexed content `x[..]` as mutable
<anon>:4     let b = &mut x[1];
                          ^~~~
error: aborting due to 2 previous errors
```

While it was plausible that borrowck could understand this simple case, it's
pretty clearly hopeless for borrowck to understand disjointness in general
container types like a tree, especially if distinct keys actually *do* map
to the same value.

In order to "teach" borrowck that what we're doing is ok, we need to drop down
to unsafe code. For instance, mutable slices expose a `split_at_mut` function that
consumes the slice and returns *two* mutable slices. One for everything to the
left of the index, and one for everything to the right. Intuitively we know this
is safe because the slices don't alias. However the implementation requires some
unsafety:

```rust,ignore
fn split_at_mut(&mut self, mid: usize) -> (&mut [T], &mut [T]) {
    unsafe {
        let self2: &mut [T] = mem::transmute_copy(&self);

        (ops::IndexMut::index_mut(self, ops::RangeTo { end: mid } ),
         ops::IndexMut::index_mut(self2, ops::RangeFrom { start: mid } ))
    }
}
```

This is pretty plainly dangerous. We use transmute to duplicate the slice with an
*unbounded* lifetime, so that it can be treated as disjoint from the other until
we unify them when we return.

However more subtle is how iterators that yield mutable references work.
The iterator trait is defined as follows:

```rust
trait Iterator {
    type Item;

    fn next(&mut self) -> Option<Self::Item>;
}
```

Given this definition, Self::Item has *no* connection to `self`. This means
that we can call `next` several times in a row, and hold onto all the results
*concurrently*. This is perfectly fine for by-value iterators, which have exactly
these semantics. It's also actually fine for shared references, as they admit
arbitrarily many references to the same thing (although the
iterator needs to be a separate object from the thing being shared). But mutable
references make this a mess. At first glance, they might seem completely
incompatible with this API, as it would produce multiple mutable references to
the same object!

However it actually *does* work, exactly because iterators are one-shot objects.
Everything an IterMut yields will be yielded *at most* once, so we don't *actually*
ever yield multiple mutable references to the same piece of data.

In general all mutable iterators require *some* unsafe code *somewhere*, though.
Whether it's raw pointers, or safely composing on top of *another* IterMut.

For instance, VecDeque's IterMut:

```rust,ignore
struct IterMut<'a, T:'a> {
    // The whole backing array. Some of these indices are initialized!
    ring: &'a mut [T],
    tail: usize,
    head: usize,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        if self.tail == self.head {
            return None;
        }
        let tail = self.tail;
        self.tail = wrap_index(self.tail.wrapping_add(1), self.ring.len());

        unsafe {
            // might as well do unchecked indexing since wrap_index has us
            // in-bounds, and many of the "middle" indices are uninitialized
            // anyway.
            let elem = self.ring.get_unchecked_mut(tail);

            // round-trip through a raw pointer to unbound the lifetime from
            // ourselves
            Some(&mut *(elem as *mut _))
        }
    }
}
```

A very subtle but interesting detail in this design is that it *relies on
privacy to be sound*. Borrowck works on some very simple rules. One of those rules
is that if we have a live &mut Foo and Foo contains an &mut Bar, then that &mut
Bar is *also* live. Since IterMut is always live when `next` can be called, if
`ring` were public then we could mutate `ring` while outstanding mutable borrows
to it exist!
