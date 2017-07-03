- Feature Name: stable_drop_order
- Start Date: 2017-01-19
- RFC PR: https://github.com/rust-lang/rfcs/pull/1857
- Rust Issue: https://github.com/rust-lang/rust/issues/43034

# Summary
[summary]: #summary

I propose we specify and stabilize drop order in Rust, instead of treating
it as an implementation detail. The stable drop order should be based on the
current implementation. This results in avoiding breakage and still allows
alternative, opt-in, drop orders to be introduced in the future.

# Motivation
[motivation]: #motivation

After lots of discussion on [issue 744](https://github.com/rust-lang/rfcs/issues/744),
there seems to be consensus about the need for a stable drop order. See, for instance,
[this](https://github.com/rust-lang/rfcs/issues/744#issuecomment-231215181) and
[this](https://github.com/rust-lang/rfcs/issues/744#issuecomment-231237499) comment.

The current drop order seems counter-intuitive (fields are dropped in FIFO order
instead of LIFO), but changing it would inevitably result in breakage. There have
been cases in the recent past when code broke because of people relying on unspecified
behavior (see for instance the
[post](https://internals.rust-lang.org/t/rolling-out-or-unrolling-struct-field-reorderings/4485)
about struct field reorderings). It is highly probable that similar breakage
would result from changes to the drop order. See for instance, the
[comment](https://github.com/rust-lang/rfcs/issues/744#issuecomment-225918642)
from @sfackler, which reflects the problems that would arise:

> Real code in the wild does rely on the current drop order, including rust-openssl,
and *there is no upgrade path* if we reverse it. Old versions of the libraries will
be subtly broken when compiled with new rustc, and new versions of the libraries
will be broken when compiled with old rustc. 

Introducing a new drop order without breaking things would require figuring out how to:

* Forbid an old compiler (with the old drop order) from compiling recent Rust
code (which could rely on the new drop order).
* Let the new compiler (with the new drop order) recognize old Rust code
(which could rely on the old drop order). This way it could choose to either:
(a) fail to compile; or (b) compile using the old drop order.

Both requirements seem quite difficult, if not impossible, to meet. Even in case
we figured out how to meet those requirements, the complexity of the approach would
probably outweight the current complexity of having a non-intuitive drop order.

Finally, in case people really dislike the current drop order, it may still
be possible to introduce alternative, opt-in, drop orders in a backwards
compatible way. However, that is not covered in this RFC.

# Detailed design
[design]: #detailed-design

The design is the same as currently implemented in rustc and is described
below. This behavior will be enforced by run-pass tests.

### Tuples, structs and enum variants

Struct fields are dropped in the same order as they are declared. Consider,
for instance, the struct below:

```rust
struct Foo {
    bar: String,
    baz: String,
}
```

In this case, `bar` will be the first field to be destroyed, followed by `baz`.

Tuples and tuple structs show the same behavior, as well as enum variants of both kinds
(struct and tuple variants).

Note that a panic during construction of one of previous data structures causes
destruction in a different order. Since the object has not yet been constructed,
its fields are treated as local variables (which are destroyed in LIFO order).
See the example below:

```rust
let x = MyStruct {
    field1: String::new(),
    field2: String::new(),
    field3: panic!()
};
```

In this case, `field2` is destructed first and `field1` second, which may
seem counterintuitive at first but makes sense when you consider that the
initialized fields are actually temporary variables. Note that the drop order
depends on the order of the fields in the *initializer* and not in the struct
declaration.

### Slices and Vec

Slices and vectors show the same behavior as structs and enums. This behavior
can be illustrated by the code below, where the first elements are dropped
first.

```rust
for x in xs { drop(x) }
``` 

If there is a panic during construction of the slice or the `Vec`, the
drop order is reversed (that is, when using `[]` literals or the `vec![]` macro).
Consider the following example:

```rust
let xs = [X, Y, panic!()];
```

Here, `Y` will be dropped first and `X` second.

### Allowed unspecified behavior

Besides the previous constructs, there are other ones that do not need
a stable drop order (at least, there is not yet evidence that it would be
useful). It is the case of `vec![expr; n]` and closure captures.

Vectors initialized with `vec![expr; n]` syntax clone the value of `expr`
in order to fill the vector. In case `clone` panics, the values produced so far
are dropped in unspecified order. The order is closely tied to an implementation
detail and the benefits of stabilizing it seem small. It is difficult to come
up with a real-world scenario where the drop order of cloned objects is relevant
to ensure some kind of invariant. Furthermore, we may want to modify the implementation
in the future.

Closure captures are also dropped in unspecified order. At this moment, it seems
like the drop order is similar to the order in which the captures are consumed within
the closure (see [this blog post](https://aochagavia.github.io/blog/exploring-rusts-unspecified-drop-order/)
for more details). Again, this order is closely tied to an implementation that
we may want to change in the future, and the benefits of stabilizing it seem small.
Furthermore, enforcing invariants through closure captures seems like a terrible footgun
at best (the same effect can be achieved with much less obscure methods, like passing
a struct as an argument).

Note: we ignore slices initialized with `[expr; n]` syntax, since they may only
contain `Copy` types, which in turn cannot implement `Drop`.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

When mentioning destructors in the Rust book, Reference and other documentation,
we should also mention the overall picture for a type that implements `Drop`.
In particular, if a `struct`/`enum` implements Drop, then when it is dropped we will
first execute the user's code and then drop all the fields (in the given order). Thus
any code in `Drop` must leave the fields in an initialized state such that they can
be dropped. If you wish to interleave the fields being dropped and user code being
executed, you can make the fields into `Option` and have a custom drop that calls take()
(or else wrap your type in a union with a single member and implement `Drop` such that
it invokes `ptr::read()` or something similar).

It is also important to mention that `union` types never drop their contents.

# Drawbacks
[drawbacks]: #drawbacks

* The counter-intuitive drop order is here to stay.

# Alternatives
[alternatives]: #alternatives

* Figure out how to let rustc know the language version targeted by a given program.
This way we could introduce a new drop order without breaking code.
* Introduce a new drop order anyway, try to minimize breakage by running crater
and hope for the best.

# Unresolved questions
[unresolved]: #unresolved-questions

* Where do we draw the line between the constructs where drop order should be stabilized
and the rest? Should the drop order of closure captures be specified? And the drop order
of `vec![expr; n]`?
