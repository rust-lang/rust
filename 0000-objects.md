- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

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

```
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

```
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


# Detailed design

To be precise about object-safety, an object-safe method:
* must not have any type parameters,
* must not take `self` by value,
* must not use `Self` (in the future, where we allow arbitrary types for the
  receiver, `Self` may only be used for the type of the receiver and only where
  we allow `Sized?` types).

A trait is object-safe if all of its methods are object-safe.

When an expression with pointer-to-concrete type is coerced to a trait object,
the compiler will check that the trait is object-safe (in addition to the usual
check that the concrete type implements the trait). It is an error for the trait
to be non-object-safe.


# Drawbacks

This is a breaking change and forbids some safe code which is legal today. This
can be addressed by splitting a trait into object-safe and non-object-safe
parts. We hope that this will lead to better design. We are not sure how much
code this will affect, it would be good to have data about this.

Example, today:

```
trait SomeTrait {
    fn foo(&self) -> int { ... }
    fn bar<U>(&self, u: Box<U>) { ... }
}

fn baz(x: &SomeTrait) {
    x.foo();
    //x.bar(box 42i);  // type error
}

```

with this proposal:

```
trait SomeTrait {
    fn foo(&self) -> int { ... }
}

trait SomeMoreTrait: SomeTrait {
    fn bar<U>(&self, u: Box<U>) { ... }
}

fn baz(x: &SomeTrait) {
    x.foo();
    //x.bar(box 42i);  // type error
}
```


# Alternatives

We could continue to check methods rather than traits are object-safe. When
checking the bounds of a type parameter for a function call where the function
is called with a trait object, we would check that all methods are object-safe
as part of the check that the actual type parameter satisfies the formal bounds.
We could probably give a different error message if the bounds are met, but the
trait is not object-safe.

Rather than the restriction on taking `self` by value, we could require a trait
is `for Sized?` in order to be object safe. The purpose of forbidding self by
value is to enforce that we always have statically known size and that we have a
vtable for dynamic dispatch. If the programmer were going to manually provide
`impl`s for each trait, we would require the `Sized?` bound on the trait to
ensure that `self` was not dereferenced. However, with the compiler-driven
approach, this is not necessary.

# Unresolved questions

N/A
