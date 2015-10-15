% References

This section gives a high-level view of the memory model that *all* Rust
programs must satisfy to be correct. Safe code is statically verified
to obey this model by the borrow checker. Unsafe code may go above
and beyond the borrow checker while still satisfying this model. The borrow
checker may also be extended to allow more programs to compile, as long as
this more fundamental model is satisfied.

There are two kinds of reference:

* Shared reference: `&`
* Mutable reference: `&mut`

Which obey the following rules:

* A reference cannot outlive its referent
* A mutable reference cannot be aliased

That's it. That's the whole model. Of course, we should probably define
what *aliased* means. To define aliasing, we must define the notion of
*paths* and *liveness*.


**NOTE: The model that follows is generally agreed to be dubious and have
issues. It's ok-ish as an intuitive model, but fails to capture the desired
semantics. We leave this here to be able to use notions introduced here in later
sections. This will be significantly changed in the future. TODO: do that.**


# Paths

If all Rust had were values (no pointers), then every value would be uniquely
owned by a variable or composite structure. From this we naturally derive a
*tree* of ownership. The stack itself is the root of the tree, with every
variable as its direct children. Each variable's direct children would be their
fields (if any), and so on.

From this view, every value in Rust has a unique *path* in the tree of
ownership. Of particular interest are *ancestors* and *descendants*: if `x` owns
`y`, then `x` is an ancestor of `y`, and `y` is a descendant of `x`. Note
that this is an inclusive relationship: `x` is a descendant and ancestor of
itself.

We can then define references as simply *names* for paths. When you create a
reference, you're declaring that an ownership path exists to this address
of memory.

Tragically, plenty of data doesn't reside on the stack, and we must also
accommodate this. Globals and thread-locals are simple enough to model as
residing at the bottom of the stack (though we must be careful with mutable
globals). Data on the heap poses a different problem.

If all Rust had on the heap was data uniquely owned by a pointer on the stack,
then we could just treat such a pointer as a struct that owns the value on the
heap. Box, Vec, String, and HashMap, are examples of types which uniquely
own data on the heap.

Unfortunately, data on the heap is not *always* uniquely owned. Rc for instance
introduces a notion of *shared* ownership. Shared ownership of a value means
there is no unique path to it. A value with no unique path limits what we can do
with it.

In general, only shared references can be created to non-unique paths. However
mechanisms which ensure mutual exclusion may establish One True Owner
temporarily, establishing a unique path to that value (and therefore all
its children). If this is done, the value may be mutated. In particular, a
mutable reference can be taken.

The most common way to establish such a path is through *interior mutability*,
in contrast to the *inherited mutability* that everything in Rust normally uses.
Cell, RefCell, Mutex, and RWLock are all examples of interior mutability types.
These types provide exclusive access through runtime restrictions.

An interesting case of this effect is Rc itself: if an Rc has refcount 1,
then it is safe to mutate or even move its internals. Note however that the
refcount itself uses interior mutability.

In order to correctly communicate to the type system that a variable or field of
a struct can have interior mutability, it must be wrapped in an UnsafeCell. This
does not in itself make it safe to perform interior mutability operations on
that value. You still must yourself ensure that mutual exclusion is upheld.




# Liveness

Note: Liveness is not the same thing as a *lifetime*, which will be explained
in detail in the next section of this chapter.

Roughly, a reference is *live* at some point in a program if it can be
dereferenced. Shared references are always live unless they are literally
unreachable (for instance, they reside in freed or leaked memory). Mutable
references can be reachable but *not* live through the process of *reborrowing*.

A mutable reference can be reborrowed to either a shared or mutable reference to
one of its descendants. A reborrowed reference will only be live again once all
reborrows derived from it expire. For instance, a mutable reference can be
reborrowed to point to a field of its referent:

```rust
let x = &mut (1, 2);
{
    // Reborrow x to a subfield.
    let y = &mut x.0;
    // y is now live, but x isn't.
    *y = 3;
}
// y goes out of scope, so x is live again.
*x = (5, 7);
```

It is also possible to reborrow into *multiple* mutable references, as long as
they are *disjoint*: no reference is an ancestor of another. Rust
explicitly enables this to be done with disjoint struct fields, because
disjointness can be statically proven:

```rust
let x = &mut (1, 2);
{
    // Reborrow x to two disjoint subfields.
    let y = &mut x.0;
    let z = &mut x.1;

    // y and z are now live, but x isn't.
    *y = 3;
    *z = 4;
}
// y and z go out of scope, so x is live again.
*x = (5, 7);
```

However it's often the case that Rust isn't sufficiently smart to prove that
multiple borrows are disjoint. *This does not mean it is fundamentally illegal
to make such a borrow*, just that Rust isn't as smart as you want.

To simplify things, we can model variables as a fake type of reference: *owned*
references. Owned references have much the same semantics as mutable references:
they can be re-borrowed in a mutable or shared manner, which makes them no
longer live. Live owned references have the unique property that they can be
moved out of (though mutable references *can* be swapped out of). This power is
only given to *live* owned references because moving its referent would of
course invalidate all outstanding references prematurely.

As a local lint against inappropriate mutation, only variables that are marked
as `mut` can be borrowed mutably.

It is interesting to note that Box behaves exactly like an owned reference. It
can be moved out of, and Rust understands it sufficiently to reason about its
paths like a normal variable.




# Aliasing

With liveness and paths defined, we can now properly define *aliasing*:

**A mutable reference is aliased if there exists another live reference to one
of its ancestors or descendants.**

(If you prefer, you may also say the two live references alias *each other*.
This has no semantic consequences, but is probably a more useful notion when
verifying the soundness of a construct.)

That's it. Super simple right? Except for the fact that it took us two pages to
define all of the terms in that definition. You know: Super. Simple.

Actually it's a bit more complicated than that. In addition to references, Rust
has *raw pointers*: `*const T` and `*mut T`. Raw pointers have no inherent
ownership or aliasing semantics. As a result, Rust makes absolutely no effort to
track that they are used correctly, and they are wildly unsafe.

**It is an open question to what degree raw pointers have alias semantics.
However it is important for these definitions to be sound that the existence of
a raw pointer does not imply some kind of live path.**
