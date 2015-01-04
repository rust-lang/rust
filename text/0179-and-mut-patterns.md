- Start Date: 23-07-2014
- RFC PR: [rust-lang/rfcs#179](https://github.com/rust-lang/rfcs/pull/179)
- Rust Issue: [rust-lang/rust#20496](https://github.com/rust-lang/rust/issues/20496)

# Summary

Change pattern matching on an `&mut T` to `&mut <pat>`, away from its
current `&<pat>` syntax.

# Motivation

Pattern matching mirrors construction for almost all types, *except*
`&mut`, which is constructed with `&mut <expr>` but destructured with
`&<pat>`. This is almost certainly an unnecessary inconsistency.

This can and does lead to confusion, since people expect the pattern
syntax to match construction, but a pattern like `&mut (ref mut x, _)` is
actually currently a parse error:

```rust
fn main() {
    let &mut (ref mut x, _);
}
```

```
and-mut-pat.rs:2:10: 2:13 error: expected identifier, found path
and-mut-pat.rs:2     let &mut (ref mut x, _);
                          ^~~
```


Another (rarer) way it can be confusing is the pattern `&mut x`. It is
expected that this binds `x` to the contents of `&mut T`
pointer... which it does, but as a mutable binding (it is parsed as
`&(mut x)`), meaning something like

```rust
for &mut x in some_iterator_over_and_mut {
    println!("{}", x)
}
```

gives an unused mutability warning. NB. it's somewhat rare that one
would want to pattern match to directly bind a name to the contents of
a `&mut` (since the normal reason to have a `&mut` is to mutate the
thing it points at, but this pattern is (byte) copying the data out,
both before and after this change), but can occur if a type only
offers a `&mut` iterator, i.e. types for which a `&` one is no more
flexible than the `&mut` one.

# Detailed design

Add `<pat> := &mut <pat>` to the pattern grammar, and require that it is used
when matching on a `&mut T`.

# Drawbacks

It makes matching through a `&mut` more verbose: `for &mut (ref mut x,
p_) in v.mut_iter()` instead of `for &(ref mut x, _) in
v.mut_iter()`.

Macros wishing to pattern match on either `&` or `&mut` need to handle
each case, rather than performing both with a single `&`. However,
macros handling these types already need special `mut` vs. not
handling if they ever name the types, or if they use `ref` vs. `ref
mut` subpatterns.

It also makes obtaining the current behaviour (binding by-value the
contents of a reference to a mutable local) slightly harder. For a
`&mut T` the pattern becomes `&mut mut x`, and, at the moment, for a
`&T`, it must be matched with `&x` and then rebound with `let mut x =
x;` (since disambiguating like `&(mut x)` doesn't yet work). However,
based on some loose grepping of the Rust repo, both of these are very
rare.

# Alternatives

None.

# Unresolved questions

None.
