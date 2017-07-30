- Feature Name: object_safe_for_dispatch
- Start Date: 2017-06-10
- RFC PR: [rust-lang/rfcs#2027](https://github.com/rust-lang/rfcs/pull/2027)
- Rust Issue: [rust-lang/rust#43561](https://github.com/rust-lang/rust/issues/43561)

# Summary
[summary]: #summary

Tweak the object safety rules to allow using trait object types for static
dispatch, even when the trait would not be safe to instantiate as an object.

# Motivation
[motivation]: #motivation

Because Rust features a very expressive type system, users often use the type
system to express high level constraints which can be resolved at compile time,
even when the types involved are never actually instantiated with values.

One common example of this is the use of "zero-sized types," or types which
contain no data. By statically dispatching over zero sized types, different
kinds of conditional or polymorphic behavior can be implemented purely at
compile time.

Another interesting case is the use of implementations on the dynamically
dispatched trait object types. Sometimes, it can be sensible to statically
dispatch different behaviors based on the name of a trait; this can be done
today by implementing traits (with only static methods) on the trait object
type:

```rust
trait Foo {
    fn foo() { }
}

trait Bar { }

// Implemented for the trait object type
impl Foo for Bar { }

fn main() {
    // Never actually instantiate a trait object:
    Bar::foo()
}
```

However, this can only be implemented if the trait being used as the receiver
is object safe. Because this behavior is entirely dispatched statically, and a
trait object is never instantiated, this restriction is not necessary. Object
safety only matters when you actually create a dynamically dispatched trait
object at runtime.

This RFC proposes to lift that restriction, allowing trait object types to be
used for static dispatch even when the trait is not object safe.

# Detailed design
[design]: #detailed-design

Today, the rules for object safey work like this:

* If the trait (e.g. `Foo`) **is** object safe:
    - The object type for the trait is a valid type.
    - The object type for the trait implements the trait; `Foo: Foo` holds.
    - Implementations of the trait can be cast to the object type; `T as Foo`
    is valid.
* If the trait (e.g. `Foo`) **is not** object safe:
    - Any attempt to use the object type for the trait is considered invalid

After this RFC, we will change the non-object-safe case to directly mirror the
object-safe case. The new rules will be:

* If the trait (e.g. `Foo`) **is not** object safe:
    - The object type for the trait **does not** implement the trait;
    `Foo: Foo` does not hold.
    - Implementations of the trait **cannot** be cast to the object type,
    `T as Foo` is not valid
    - **However**, the object type is still a valid type. It just does not meet
    the self-trait bound, and it cannot be instantiated in safe Rust.

This change to the rules will allow trait object types to be used for static
dispatch.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

This is just a slight tweak to how object safety is implemented. We will need
to make sure the the official documentation is accurate to the rules,
especially the reference.

However, this does not need to be **highlighted** to users per se in the
explanation of object safety. This tweak will only impact advanced uses of the
trait system.

# Drawbacks
[drawbacks]: #drawbacks

This is a change to an existing system, its always possible it could cause
regressions, though the RFC authors are unaware of any.

Arguably, the rules become more nuanced (though they also become a more direct
mirror).

This would allow instantiating object types for non-object safe traits in
unsafe code, by transmuting from `std::raw::TraitObject`. This would be
extremely unsafe and users almost certainly should not do this. In the status
quo, they just can't.

# Alternatives
[alternatives]: #alternatives

We could instead make it possible for every trait to be object safe, by
allowing `where Self: Sized` bounds on every single item. For example:

```rust
// Object safe because all of these non-object safe items are constrained
// `Self: Sized.`
trait Foo {
    const BAR: usize where Self: Sized;
    type Baz where Self: Sized;
    fn quux() where Self: Sized;
    fn spam<T: Eggs>(&self) where Self: Sized;
}
```

However, this puts the burden on users to add all of these additional bounds.

Possibly we should add bounds like this in addition to this RFC, since they
are already valid on functions, just not types and consts.

# Unresolved questions
[unresolved]: #unresolved-questions

How does this impact the implementation in rustc?
