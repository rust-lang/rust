# promotable_const_fn

The tracking issue for this feature is: None.

The `#[promotable_const_fn]` attribute can be used if the `promotable_const_fn` feature gate is
active.
While there are no checks on where it can be used, it only has an effect if applied to a `const fn`.
Since this attribute is never meant to be stabilized, this seems like an ok footgun.

The attribute exists, so it is possible to stabilize the constness of a function without stabilizing
the promotablity of the function at the same time. We eventually want to have a better strategy,
but we need to hash out the details. Until then, any const fn that is doing anything nontrivial
should not be marked with `#[promotable_const_fn]`. Even better: do not mark any functions with the
attribute. The existing "promotable on stable" functions were marked, but we could simply not
stabilize the promotability of any further functions until we have figured out a clean solution
for all the issues that come with promotion.

The reason this attribute even exists are functions like

```rust,ignore
#![feature(const_fn)]

fn main() {
    let x: &'static bool = &foo(&1, &2);
}

union Foo<'a> {
    a: &'a u8,
    b: usize,
}

const fn foo(a: &u8, b: &u8) -> bool {
    unsafe { Foo { a: a }.b == Foo { a: b }.b }
}
```

Where this would be perfectly fine at runtime, but changing the function to a const fn would cause
it to be evaluated at compile-time and thus produce a lint about not being able to compute the value
and then emitting a llvm `undef` because there is no value to place into the promoted.

This attribute patches this hole by not promoting functions without the attribute and at the same
time preventing functions with the attribute from using unions or calling other functions that do
not have said attribute.