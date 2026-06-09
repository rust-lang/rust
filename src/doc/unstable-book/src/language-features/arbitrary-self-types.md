# `arbitrary_self_types`

The tracking issue for this feature is: [#44874]

[#44874]: https://github.com/rust-lang/rust/issues/44874

------------------------

Allows any type implementing `core::ops::Receiver<Target=T>` to be used as the type
of `self` in a method belonging to `T`.

For example,

```rust
#![feature(arbitrary_self_types)]

struct A;

impl A {
    fn f(self: SmartPtr<Self>) -> i32 { 1 }  // note self type
}

struct SmartPtr<T>(T);

impl<T> core::ops::Receiver for SmartPtr<T> {
    type Target = T;
}

fn main() {
    let smart_ptr = SmartPtr(A);
    assert_eq!(smart_ptr.f(), 1);
}
```

The `Receiver` trait has a blanket implementation for all `T: Deref`, so in fact
things like this work too:

```rust
#![feature(arbitrary_self_types)]

use std::rc::Rc;

struct A;

impl A {
    fn f(self: Rc<Self>) -> i32 { 1 } // Rc implements Deref
}

fn main() {
    let smart_ptr = Rc::new(A);
    assert_eq!(smart_ptr.f(), 1);
}
```

Interestingly, that works even without the `arbitrary_self_types` feature
- but that's because certain types are _effectively_ hard coded, including
`Rc`. ("Hard coding" isn't quite true; they use a lang-item called
`LegacyReceiver` to denote their special-ness in this way). With the
`arbitrary_self_types` feature, their special-ness goes away, and custom
smart pointers can achieve the same.

## Changes to method lookup

Method lookup previously used to work by stepping through the `Deref`
chain then using the resulting list of steps in two different ways:

* To identify types that might contribute methods via their `impl`
  blocks (inherent methods) or via traits
* To identify the types that the method receiver (`a` in the above
  examples) can be converted to.

With this feature, these lists are created by instead stepping through
the `Receiver` chain. However, a note is kept about whether the type
can be reached also via the `Deref` chain.

The full chain (via `Receiver` hops) is used for the first purpose
(identifying relevant `impl` blocks and traits); whereas the shorter
list (reachable via `Deref`) is used for the second purpose. That's
because, to convert the method target (`a` in `a.b()`) to the self
type, Rust may need to be able to use `Deref::deref`. Type conversions,
then, can only proceed as far as the end of the `Deref` chain whereas
the longer `Receiver` chain can be used to explore more places where
useful methods might reside.

## Types suitable for use as smart pointers

This feature allows the creation of customised smart pointers - for example
your own equivalent to `Rc` or `Box` with whatever capabilities you like.
Those smart pointers can either implement `Deref` (if it's safe to
create a reference to the referent) or `Receiver` (if it isn't).

Either way, smart pointer types should mostly _avoid having methods_.
Calling methods on a smart pointer leads to ambiguity about whether you're
aiming for a method on the pointer, or on the referent.

Best practice is therefore to put smart pointer functionality into
associated functions instead - that's what's done in all the smart pointer
types within Rust's standard library which implement `Receiver`.

If you choose to add any methods to your smart pointer type, your users
may run into errors from deshadowing, as described in the next section.

## Avoiding shadowing

With or without this feature, Rust emits an error if it finds two method
candidates, like this:

```rust,compile_fail
use std::pin::Pin;
use std::pin::pin;

struct A;

impl A {
    fn get_ref(self: Pin<&A>) {}
}

fn main() {
    let pinned_a: Pin<&A> = pin!(A).as_ref();
    let pinned_a: Pin<&A> = pinned_a.as_ref();
    pinned_a.get_ref(); // error[E0034]: multiple applicable items in scope
}
```

(this is why Rust's smart pointers are mostly carefully designed to avoid
having methods at all, and shouldn't add new methods in future.)

With `arbitrary_self_types`, we take care to spot some other kinds of
conflict:

```rust,compile_fail
#![feature(arbitrary_self_types)]

use std::pin::Pin;
use std::pin::pin;

struct A;

impl A {
    fn get_ref(self: &Pin<&A>) {}  // note &Pin
}

fn main() {
    let pinned_a: Pin<&mut A> = pin!(A);
    let pinned_a: Pin<&A> = pinned_a.as_ref();
    pinned_a.get_ref();
}
```

This is to guard against the case where an inner (referent) type has a
method of a given name, taking the smart pointer by reference, and then
the smart pointer implementer adds a similar method taking self by value.
As noted in the previous section, the safe option is simply
not to add methods to smart pointers, and then these errors can't occur.
