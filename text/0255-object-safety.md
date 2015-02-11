- Start Date: 2014-09-22
- RFC PR: [rust-lang/rfcs#255](https://github.com/rust-lang/rfcs/pull/255)
- Rust Issue: [rust-lang/rust#17670](https://github.com/rust-lang/rust/issues/17670)

# Summary

Restrict which traits can be used to make trait objects.

Currently, we allow any traits to be used for trait objects, but restrict the
methods which can be called on such objects. Here, we propose instead
restricting which traits can be used to make objects. Despite being less
flexible, this will make for better error messages, less surprising software
evolution, and (hopefully) better design. The motivation for the proposed change
is stronger due to part of the DST changes.

# Motivation

Part of the planned, in progress DST work is to allow trait objects where a
trait is expected. Example:

```rust
fn foo<Sized? T: SomeTrait>(y: &T) { ... }

fn bar(x: &SomeTrait) {
    foo(x)
}
```

Previous to DST the call to `foo` was not expected to work because `SomeTrait`
was not a type, so it could not instantiate `T`. With DST this is possible, and
it makes intuitive sense for this to work (an alternative is to require `impl
SomeTrait for SomeTrait { ... }`, but that seems weird and confusing and rather
like boilerplate. Note that the precise mechanism here is out of scope for this
RFC).

This is only sound if the trait is /object-safe/. We say a method `m` on trait
`T` is object-safe if it is legal (in current Rust) to call `x.m(...)` where `x`
has type `&T`, i.e., `x` is a trait object. If all methods in `T` are object-
safe, then we say `T` is object-safe.

If we ignore this restriction we could allow code such as the following:

```rust
trait SomeTrait {
    fn foo(&self, other: &Self) { ... } // assume self and other have the same concrete type
}

fn bar<Sized? T: SomeTrait>(x: &T, y: &T) {
    x.foo(y); // x and y may have different concrete types, pre-DST we could
        // assume that x and y had the same concrete types.
}

fn baz(x: &SomeTrait, y: &SomeTrait) {
    bar(x, y) // x and y may have different concrete types
}
```

This RFC proposes enforcing object-safety when trait objects are created, rather
than where methods on a trait object are called or where we attempt to match
traits. This makes both method call and using trait objects with generic code
simpler. The downside is that it makes Rust less flexible, since not all traits
can be used to create trait objects.

Software evolution is improved with this proposal: imagine adding a non-object-
safe method to a previously object-safe trait. With this proposal, you would
then get errors wherever a trait-object is created. The error would explain why
the trait object could not be created and point out exactly which method was to
blame and why. Without this proposal, the only errors you would get would be
where a trait object is used with a generic call and would be something like
"type error: SomeTrait does not implement SomeTrait" - no indication that the
non-object-safe method were to blame, only a failure in trait matching.

Another advantage of this proposal is that it implies that all
method-calls can always be rewritten into an equivalent [UFCS]
call. This simplifies the "core language" and makes method dispatch
notation -- which involves some non-trivial inference -- into a kind
of "sugar" for the more explicit UFCS notation.

# Detailed design

To be precise about object-safety, an object-safe method must meet one
of the following conditions:

* require `Self : Sized`; or,
* meet all of the following conditions:
  * must not have any type parameters; and,
  * must have a receiver that has type `Self` or which dereferences to the `Self` type;
    - for now, this means `self`, `&self`, `&mut self`, or `self: Box<Self>`,
      but eventually this should be extended to custom types like
      `self: Rc<Self>` and so forth.
  * must not use `Self` (in the future, where we allow arbitrary types
    for the receiver, `Self` may only be used for the type of the
    receiver and only where we allow `Sized?` types).

A trait is object-safe if all of the following conditions hold:

* all of its methods are object-safe; and,
* the trait does not require that `Self : Sized` (see also [RFC 546]).

When an expression with pointer-to-concrete type is coerced to a trait object,
the compiler will check that the trait is object-safe (in addition to the usual
check that the concrete type implements the trait). It is an error for the trait
to be non-object-safe.

Note that a trait can be object-safe even if some of its methods use
features that are not supported with an object receiver. This is true
when code that attempted to use those features would only work if the
`Self` type is `Sized`. This is why all methods that require
`Self:Sized` are exempt from the typical rules. This is also why
by-value self methods are permitted, since currently one cannot invoke
pass an unsized type by-value (though we consider that a useful future
extension).

# Drawbacks

This is a breaking change and forbids some safe code which is legal
today. This can be addressed in two ways: splitting traits, or adding
`where Self:Sized` clauses to methods that cannot not be used with
objects.

### Example problem

Here is an example trait that is not object safe:

```rust
trait SomeTrait {
    fn foo(&self) -> int { ... }
    
    // Object-safe methods may not return `Self`:
    fn new() -> Self;
}
```

### Splitting a trait

One option is to split a trait into object-safe and non-object-safe
parts. We hope that this will lead to better design. We are not sure
how much code this will affect, it would be good to have data about
this.

```rust
trait SomeTrait {
    fn foo(&self) -> int { ... }
}

trait SomeTraitCtor : SomeTrait {
    fn new() -> Self;
}
```

### Adding a where-clause

Sometimes adding a second trait feels like overkill. In that case, it
is often an option to simply add a `where Self:Sized` clause to the
methods of the trait that would otherwise violate the object safety
rule.

```rust
trait SomeTrait {
    fn foo(&self) -> int { ... }
    
    fn new() -> Self
        where Self : Sized; // this condition is new
}
```

The reason that this makes sense is that if one were writing a generic
function with a type parameter `T` that may range over the trait
object, that type parameter would have to be declared `?Sized`, and
hence would not have access to the `new` method:

```rust
fn baz<T:?Sized+SomeTrait>(t: &T) {
    let v: T = SomeTrait::new(); // illegal because `T : Sized` is not known to hold
}
```

However, if one writes a function with sized type parameter, which
could never be a trait object, then the `new` function becomes
available.

```rust
fn baz<T:SomeTrait>(t: &T) {
    let v: T = SomeTrait::new(); // OK
}
```

# Alternatives

We could continue to check methods rather than traits are
object-safe. When checking the bounds of a type parameter for a
function call where the function is called with a trait object, we
would check that all methods are object-safe as part of the check that
the actual type parameter satisfies the formal bounds.  We could
probably give a different error message if the bounds are met, but the
trait is not object-safe.

We might in the future use finer-grained reasoning to permit more
non-object-safe methods from appearing in the trait. For example, we
might permit `fn foo() -> Self` because it (implicitly) requires that
`Self` be sized. Similarly, we might permit other tests beyond just
sized-ness. Any such extension would be backwards compatible.

# Unresolved questions

N/A

# Edits

* 2014-02-09. Edited by Nicholas Matsakis to (1) include the
  requirement that object-safe traits do not require `Self:Sized` and
  (2) specify that methods may include `where Self:Sized` to overcome
  object safety restrictions.

[UFCS]: 0132-ufcs.md
[RFC 546]: 0546-Self-not-sized-by-default.md
