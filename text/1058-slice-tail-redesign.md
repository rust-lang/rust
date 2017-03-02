- Feature Name: `slice_tail_redesign`
- Start Date: 2015-04-11
- RFC PR: [rust-lang/rfcs#1058](https://github.com/rust-lang/rfcs/pull/1058)
- Rust Issue: [rust-lang/rust#26906](https://github.com/rust-lang/rust/issues/26906)

# Summary

Replace `slice.tail()`, `slice.init()` with new methods `slice.split_first()`,
`slice.split_last()`.

# Motivation

The `slice.tail()` and `slice.init()` methods are relics from an older version
of the slice APIs that included a `head()` method. `slice` no longer has
`head()`, instead it has `first()` which returns an `Option`, and `last()` also
returns an `Option`. While it's generally accepted that indexing / slicing
should panic on out-of-bounds access, `tail()`/`init()` are the only
remaining methods that panic without taking an explicit index.

A conservative change here would be to simply change `head()`/`tail()` to return
`Option`, but I believe we can do better. These operations are actually
specializations of `split_at()` and should be replaced with methods that return
`Option<(&T,&[T])>`. This makes the common operation of processing the
first/last element and the remainder of the list more ergonomic, with very low
impact on code that only wants the remainder (such code only has to add `.1` to
the expression). This has an even more significant effect on code that uses the
mutable variants.

# Detailed design

The methods `head()`, `tail()`, `head_mut()`, and `tail_mut()` will be removed,
and new methods will be added:

```rust
fn split_first(&self) -> Option<(&T, &[T])>;
fn split_last(&self) -> Option<(&T, &[T])>;
fn split_first_mut(&mut self) -> Option<(&mut T, &mut [T])>;
fn split_last_mut(&mut self) -> Option<(&mut T, &mut [T])>;
```

Existing code using `tail()` or `init()` could be translated as follows:

* `slice.tail()` becomes `&slice[1..]`
* `slice.init()` becomes `&slice[..slice.len()-1]` or
  `slice.split_last().unwrap().1`

It is expected that a lot of code using `tail()` or `init()` is already either
testing `len()` explicitly or using `first()` / `last()` and could be refactored
to use `split_first()` / `split_last()` in a more ergonomic fashion. As an
example, the following code from typeck:

```rust
if variant.fields.len() > 0 {
    for field in variant.fields.init() {
```

can be rewritten as:

```rust
if let Some((_, init_fields)) = variant.fields.split_last() {
    for field in init_fields {
```

And the following code from compiletest:

```rust
let argv0 = args[0].clone();
let args_ = args.tail();
```

can be rewritten as:

```rust
let (argv0, args_) = args.split_first().unwrap();
```

(the `clone()` ended up being unnecessary).

# Drawbacks

The expression `slice.split_last().unwrap().1` is more cumbersome than
`slice.init()`. However, this is primarily due to the need for `.unwrap()`
rather than the need for `.1`, and would affect the more conservative solution
(of making the return type `Option<&[T]>`) as well. Furthermore, the more
idiomatic translation is `&slice[..slice.len()-1]`, which can be used any time
the slice is already stored in a local variable.

# Alternatives

Only change the return type to `Option` without adding the tuple. This is the
more conservative change mentioned above. It still has the same drawback of
requiring `.unwrap()` when translating existing code. And it's unclear what the
function names should be (the current names are considered suboptimal).

Just deprecate the current methods without adding replacements. This gets rid of
the odd methods today, but it doesn't do anything to make it easier to safely
perform these operations.
