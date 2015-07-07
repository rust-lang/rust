% References

There are two kinds of reference:

* Shared reference: `&`
* Mutable reference: `&mut`

Which obey the following rules:

* A reference cannot outlive its referent
* A mutable reference cannot be aliased

To define aliasing, we must define the notion of *paths* and *liveness*.




# Paths

If all Rust had were values, then every value would be uniquely owned
by a variable or composite structure. From this we naturally derive a *tree*
of ownership. The stack itself is the root of the tree, with every variable
as its direct children. Each variable's direct children would be their fields
(if any), and so on.

From this view, every value in Rust has a unique *path* in the tree of ownership.
References to a value can subsequently be interpreted as a path in this tree.
Of particular interest are *prefixes*: `x` is a prefix of `y` if `x` owns `y`

However much data doesn't reside on the stack, and we must also accommodate this.
Globals and thread-locals are simple enough to model as residing at the bottom
of the stack (though we must be careful with mutable globals). Data on
the heap poses a different problem.

If all Rust had on the heap was data uniquely by a pointer on the stack,
then we can just treat that pointer as a struct that owns the value on
the heap. Box, Vec, String, and HashMap, are examples of types which uniquely
own data on the heap.

Unfortunately, data on the heap is not *always* uniquely owned. Rc for instance
introduces a notion of *shared* ownership. Shared ownership means there is no
unique path. A value with no unique path limits what we can do with it. In general, only
shared references can be created to these values. However mechanisms which ensure
mutual exclusion may establish One True Owner temporarily, establishing a unique path
to that value (and therefore all its children).

The most common way to establish such a path is through *interior mutability*,
in contrast to the *inherited mutability* that everything in Rust normally uses.
Cell, RefCell, Mutex, and RWLock are all examples of interior mutability types. These
types provide exclusive access through runtime restrictions. However it is also
possible to establish unique ownership without interior mutability. For instance,
if an Rc has refcount 1, then it is safe to mutate or move its internals.




# Liveness

Roughly, a reference is *live* at some point in a program if it can be
dereferenced. Shared references are always live unless they are literally unreachable
(for instance, they reside in freed or leaked memory). Mutable references can be
reachable but *not* live through the process of *reborrowing*.

A mutable reference can be reborrowed to either a shared or mutable reference.
Further, the reborrow can produce exactly the same reference, or point to a
path it is a prefix of. For instance, a mutable reference can be reborrowed
to point to a field of its referent:

```rust
let x = &mut (1, 2);
{
    // reborrow x to a subfield
    let y = &mut x.0;
    // y is now live, but x isn't
    *y = 3;
}
// y goes out of scope, so x is live again
*x = (5, 7);
```

It is also possible to reborrow into *multiple* mutable references, as long as
they are *disjoint*: no reference is a prefix of another. Rust
explicitly enables this to be done with disjoint struct fields, because
disjointness can be statically proven:

```rust
let x = &mut (1, 2);
{
    // reborrow x to two disjoint subfields
    let y = &mut x.0;
    let z = &mut x.1;
    // y and z are now live, but x isn't
    *y = 3;
    *z = 4;
}
// y and z go out of scope, so x is live again
*x = (5, 7);
```

However it's often the case that Rust isn't sufficiently smart to prove that
multiple borrows are disjoint. *This does not mean it is fundamentally illegal
to make such a borrow*, just that Rust isn't as smart as you want.

To simplify things, we can model variables as a fake type of reference: *owned*
references. Owned references have much the same semantics as mutable references:
they can be re-borrowed in a mutable or shared manner, which makes them no longer
live. Live owned references have the unique property that they can be moved
out of (though mutable references *can* be swapped out of). This is
only given to *live* owned references because moving its referent would of
course invalidate all outstanding references prematurely.

As a local lint against inappropriate mutation, only variables that are marked
as `mut` can be borrowed mutably.

It is also interesting to note that Box behaves exactly like an owned
reference. It can be moved out of, and Rust understands it sufficiently to
reason about its paths like a normal variable.




# Aliasing

With liveness and paths defined, we can now properly define *aliasing*:

**A mutable reference is aliased if there exists another live reference to it or
one of its prefixes.**

That's it. Super simple right? Except for the fact that it took us two pages
to define all of the terms in that defintion. You know: Super. Simple.

Actually it's a bit more complicated than that. In addition to references,
Rust has *raw pointers*: `*const T` and `*mut T`. Raw pointers have no inherent
ownership or aliasing semantics. As a result, Rust makes absolutely no effort
to track that they are used correctly, and they are wildly unsafe.

**It is an open question to what degree raw pointers have alias semantics.
However it is important for these definitions to be sound that the existence
of a raw pointer does not imply some kind of live path.**
