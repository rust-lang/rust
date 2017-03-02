- Start Date: 2015-01-26
- RFC PR: https://github.com/rust-lang/rfcs/pull/736
- Rust Issue: https://github.com/rust-lang/rust/issues/21407

# Summary

Change Functional Record Update (FRU) for struct literal expressions
to respect struct privacy.

# Motivation

Functional Record Update is the name for the idiom by which one can
write `..<expr>` at the end of a struct literal expression to fill in
all remaining fields of the struct literal by using `<expr>` as the
source for them.

```rust
mod foo {
    pub struct Bar { pub a: u8, pub b: String, _cannot_construct: () }

    pub fn new_bar(a: u8, b: String) -> Bar {
        Bar { a: a, b: b, _cannot_construct: () }
    }
}

fn main() {
    let bar_1 = foo::new_bar(3, format!("bar one"));

    let bar_2a = foo::Bar { b: format!("bar two"), ..bar_1 }; // FRU!

    println!("bar_1: {} bar_2a: {}", bar_1.b, bar_2a.b);

    let bar_2b = foo::Bar { a: 17, ..bar_2a };                // FRU again!

    println!("bar_1: {} bar_2b: {}", bar_1.b, bar_2b.b);
}
```

Currently, Functional Record Update will freely move or copy all
fields not explicitly mentioned in the struct literal expression,
so the code above runs successfully.

In particular, consider a case like this:

```rust
#![allow(unstable)]
extern crate alloc;
use self::foo::Secrets;
mod foo {
    use alloc;
    #[allow(raw_pointer_derive)]
    #[derive(Debug)]
    pub struct Secrets { pub a: u8, pub b: String, ptr: *mut u8 }

    pub fn make_secrets(a: u8, b: String) -> Secrets {
        let ptr = unsafe { alloc::heap::allocate(10, 1) };
        Secrets { a: a, b: b, ptr: ptr }
    }

    impl Drop for Secrets {
        fn drop(&mut self) {
            println!("because of {}, deallocating {:p}", self.b, self.ptr);
            unsafe { alloc::heap::deallocate(self.ptr, 10, 1); }
        }
    }
}

fn main() {
    let s_1 = foo::make_secrets(3, format!("ess one"));
    let s_2 = foo::Secrets { b: format!("ess two"), ..s_1 }; // FRU ...

    println!("s_1.b: {} s_2.b: {}", s_1.b, s_2.b);
    // at end of scope, ... both s_1 *and* s_2 get dropped.  Boom!
}
```

This example prints the following (if one's memory allocator is not checking for double-frees):

```text
s_1.b: ess one s_2.b: ess two
because of ess two, deallocating 0x7f00c182e000
because of ess one, deallocating 0x7f00c182e000
```

In particular, from reading the module `foo`, it appears that one is
attempting to preserve an invariant that each instance of `Secrets`
has its own unique `ptr` value; but this invariant is broken by the use
of FRU.

Note that there is essentially no way around this abstraction
violation today; as shown for example in [Issue 21407], where
the backing storage for a `Vec` is duplicated in a second `Vec`
by use of the trivial FRU expression `{ ..t }` where `t: Vec<T>`.

[Issue 21407]: https://github.com/rust-lang/rust/issues/21407#issuecomment-71374092

Again, this is due to the current rule that Functional Record Update
will freely move or copy all fields not explicitly mentioned in the
struct literal expression, *regardless* of whether they are visible
(in terms of privacy) in the spot in code.

This RFC proposes to change that rule, and say that a struct literal
expression using FRU is effectively expanded into a complete struct
literal with initializers for all fields (i.e., a struct literal that
does not use FRU), and that this expanded struct literal is subject to
privacy restrictions.

The main motivation for this is to plug this abstraction-violating
hole with as little other change to the rules, implementation, and
character of the Rust language as possible.


# Detailed design

As already stated above, the change proposed here is that a struct
literal expression using FRU is effectively expanded into a complete
struct literal with initializers for all fields (i.e., a struct
literal that does not use FRU), and that this expanded struct literal
is subject to privacy restrictions.

(Another way to think of this change is: one can only use FRU with a
struct if one has visibility of all of its declared fields. If any
fields are hidden by privacy, then all forms of struct literal syntax
are unavailable, including FRU.)

----

This way, the `Secrets` example above will be essentially equivalent to
```rust
#![allow(unstable)]
extern crate alloc;
use self::foo::Secrets;
mod foo {
    use alloc;
    #[allow(raw_pointer_derive)]
    #[derive(Debug)]
    pub struct Secrets { pub a: u8, pub b: String, ptr: *mut u8 }

    pub fn make_secrets(a: u8, b: String) -> Secrets {
        let ptr = unsafe { alloc::heap::allocate(10, 1) };
        Secrets { a: a, b: b, ptr: ptr }
    }

    impl Drop for Secrets {
        fn drop(&mut self) {
            println!("because of {}, deallocating {:p}", self.b, self.ptr);
            unsafe { alloc::heap::deallocate(self.ptr, 10, 1); }
        }
    }
}

fn main() {
    let s_1 = foo::make_secrets(3, format!("ess one"));
    // let s_2 = foo::Secrets { b: format!("ess two"), ..s_1 };
    // is rewritten to:
    let s_2 = foo::Secrets { b: format!("ess two"),
                             /* remainder from FRU */
                             a: s_1.a, ptr: s_1.ptr };

    println!("s_1.b: {} s_2.b: {}", s_1.b, s_2.b);
}
```

which is rejected as field `ptr` of `foo::Secrets` is private and
cannot be accessed from `fn main` (both in terms of reading it from
`s_1`, but also in terms of using it to build a new instance of
`foo::Secrets`.

----

(While the change to the language is described above in terms of
rewriting the code, the implementation need not go that route. In
particular, [this commit] shows a different strategy that is isolated
to the `librustc_privacy` crate.)

[this commit]: https://github.com/pnkfelix/rust/commit/c651bac4189dc03d6a5637323b6ae02fc30e711a

----

The proposed change is applied only to struct literal expressions.  In
particular, enum struct variants are left unchanged, since all of
their fields are already implicitly public.

# Drawbacks

There is a use case for allowing private fields to be moved/copied via
FRU, which I call the "future extensibility" library design pattern:
it is a convenient way for a library author to tell clients to make
updated copies of a record in a manner that is oblivious to the
addition of new private fields to the struct (at least, new private
fields that implement `Copy`...).

For example, in Rust today without the change proposed here, in the
first example above using `Bar`, the author of the `mod foo` can
change `Bar` like so:

```rust
    pub struct Bar { pub a: u8, pub b: String, _hidden: u8 }

    pub fn new_bar(a: u8, b: String) -> Bar {
        Bar { a: a, b: b, _hidden: 17 }
    }
```

And all of the code from the `fn main` in the first example will
continue to run.

Also, when the struct is moved (rather than copied) by the FRU
expression, the same pattern applies and works even when the new
private fields do not implement `Copy`.

However, there is a small coding pattern that enables such continued
future-extensibility for library authors: divide the struct into the
entirely `pub` frontend, with one member that is the `pub` backend
with entirely private contents, like so:

```rust
mod foo {
    pub struct Bar { pub a: u8, pub b: String, pub _hidden: BarHidden }
    pub struct BarHidden { _cannot_construct: () }
    fn new_hidden() -> BarHidden {
        BarHidden { _cannot_construct: () }
    }

    pub fn new_bar(a: u8, b: String) -> Bar {
        Bar { a: a, b: b, _hidden: new_hidden() }
    }
}

fn main() {
    let bar_1 = foo::new_bar(3, format!("bar one"));

    let bar_2a = foo::Bar { b: format!("bar two"), ..bar_1 }; // FRU!

    println!("bar_1: {} bar_2a: {}", bar_1.b, bar_2a.b);

    let bar_2b = foo::Bar { a: 17, ..bar_2a };                // FRU again!

    println!("bar_1: {} bar_2b: {}", bar_1.b, bar_2b.b);
}
```

All hidden changes that one would have formerly made to `Bar` itself
are now made to `BarHidden`.  The struct `Bar` is entirely public (including
the supposedly-hidden field named `_hidden`), and
thus can be legally be used with FRU in all client contexts that can
see the type `Bar`, even under the new rules proposed by this RFC.



# Alternatives

Most Important: If we do not do *something* about this, then both stdlib types like
`Vec` and user-defined types will fundmentally be unable to enforce
abstraction. In other words, the Rust language will be broken.

----

glaebhoerl and pnkfelix outlined a series of potential alternatives, including this one.
Here is an attempt to transcribe/summarize them:

  1. Change the FRU form `Bar { x: new_x, y: new_y, ..old_b }` so it
     somehow is treated as consuming `old_b`, rather than
     moving/copying each of the remaining fields in `old_b`.

     It is not totally clear what the semantics actually are for this
     form. Also, there may not be time to do this properly for 1.0.

  2. Try to adopt a data/abstract-type distinction along the lines of the one in [glaebhoerl's draft RFC]. 

[glaebhoerl's draft RFC]: https://raw.githubusercontent.com/glaebhoerl/rust-notes/master/my_rfcs/Distinguish%20data%20types%20from%20abstract%20types.txt

     As a special subnote on this alternative: While [glaebhoerl's draft RFC] proposed
     syntactic forms for indicating the data/abstract-type distinction, we could
     also (or instead) do it based solely on the presence of a single non-`pub`
     field, as pointed out by glaebhoerl at the [comment here].

[comment here]: https://github.com/rust-lang/rust/issues/21407#issuecomment-71196581

    (Another potential criterion could be "has *all* private fields."; see
     related discussion below in the item "Outlaw the trivial FRU form Foo".)

  3. let FRU keep its current privacy violating semantics, but also
     make FRU something one must opt-in to support on a type. E.g. make
     a builtin `FunUpdate` trait that a struct must implement in order
     to be usable with FRU. (Or maybe its an attribute you attach to
     the struct item.)

     This approach would impose a burden on all code today that makes
     use of FRU, since they would have to start implementing
     `FunUpdate`. Thus, not simple to implement for the libraries and
     the overall ecosystem.  What other designs have been considered?
     What is the impact of not doing this?

  4. Adopt this RFC, but add a builtin `HygienicFunUpdate` trait that
     one can opt-into to get the old (privacy violating) semantics.

     While this is obviously complicated, it has the advantage that it
     has a staged landing strategy: We could just adopt and implement
     this RFC for 1.0 beta. We could add `HygienicFunUpdate` at an
     arbitrary point in the future; it would not have to be in the 1.0
     release.

     (For why the trait is named `HygienicFunUpdate`, see comment
      thread on [Issue 21407].)

  5. Add way for struct item to opt out of FRU support entirely,
     e.g. via an attribute.

     This seems pretty fragile; i.e., easy to forget.

  6. Outlaw the trivial FRU form Foo { ..<expr> }. That is, to use
     FRU, you have to use at least one field in the constructing
     expression. Again, this implies that types like Vec and HashMap
     will not be subject to the vulnerability outlined here.

     This solves the vulnerability for types like `Vec` and `HashMap`,
     but the `Secrets` example from the Motivation section still
     breaks; the author for the `mod foo` library will need to write
     their code more carefully to ensure that secret things are
     contained in a separate struct with all private fields,
     much like the `BarHidden` code pattern discussed above.

# Unresolved questions

How important is the "future extensibility" library design pattern
described in the Drawbacks section?  How many Cargo packages, if any,
use it?
