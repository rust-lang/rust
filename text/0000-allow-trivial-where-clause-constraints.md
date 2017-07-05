- Feature Name: `allow_trivial_constraints`
- Start Date: 2017-07-05
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Allow constraints to appear in where clauses which are trivially known to either
always hold or never hold. This would mean that `impl Foo for Bar where i32:
Iterator` would become valid, and the impl would never be satisfied.

# Motivation
[motivation]: #motivation

It may seem strange to ever want to include a constraint that is always known to
hold or not hold. However, as with many of these cases, allowing this would be
useful for macros. For example, a custom derive may want to add additional
functionality if two derives are used together. As another more concrete
example, Diesel allows the use of normal Rust operators to generate the
equivalent SQL. Due to coherence rules, we can't actually provide a blanket
impl, but we'd like to automatically implement `std::ops::Add` for columns when
they are of a type for which `+` is a valid operator. The generated impl would
look like:

```rust
impl<T> std::ops::Add<T> for my_column
where
    my_column::SqlType: diesel::types::ops::Add,
    T: AsExpression<<my_column::SqlType as diesel::types::ops::Add>::Rhs>,
{
    // ...
}
```

One would never write this impl normally since we always know the type of
`my_column::SqlType`. However, when you consider the use case of a macro, we
can't always easily know whether that constraint would hold or not at the time
when we're generating code.

# Detailed design
[design]: #detailed-design

Concretely implementing this means the removal of [`E0193`]. Interestingly, as of
Rust 1.7, that error never actually appears. Instead the current behavior is
that something like `impl Foo for Bar where i32: Copy` (e.g. anywhere that the
constraint always holds) compiles fine, and `impl Foo for Bar where i32:
Iterator` fails to compile by complaining that `i32` does not implement
`Iterator`. The original error message explicitly forbidding this case does not
seem to ever appear.

The obvious complication that comes to mind when implementing this feature is
that it would allow nonsensical projections to appear in the where clause as
well. For example, when `i32: IntoIterator` appears in a where clause, we would
also need to allow `i32::Item: SomeTrait` to appear in the same clause, and even
allow `for _ in 1` to appear in item bodies, and have it all successfully
compile.

[`E0193`]: https://doc.rust-lang.org/error-index.html#E0193

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

This feature does not need to be taught explicitly. Knowing the basic rules of
where clauses, one would naturally already expect this to work.

# Drawbacks
[drawbacks]: #drawbacks

- Code that is pretty obviously nonsense outside of the context of a macro or
  derive would become valid.
- The changes to the compiler could potentially increase complexity quite a bit

# Alternatives
[alternatives]: #alternatives

n/a

# Unresolved questions
[unresolved]: #unresolved-questions

n/a
