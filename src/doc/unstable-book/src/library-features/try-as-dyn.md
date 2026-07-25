# `try_as_dyn`

The tracking issue for this feature is: [#144361]

[#144361]: https://github.com/rust-lang/rust/issues/144361

------------------------

The `try_as_dyn` feature allows going from a generic `T` with no bounds
to a `dyn Trait`, if `T: Trait` and various conditions are upheld. It is
very related to specialization, as it allows you to specialize within
function bodies, but in a more general manner than `Any::downcast`.

```rust
#![feature(try_as_dyn)]

fn downcast_debug_format<T: 'static>(t: &T) -> String {
    match std::any::try_as_dyn::<_, dyn std::fmt::Debug>(t) {
        Some(d) => format!("{d:?}"),
        None => "default".to_string()
    }
}
```


## Rules and reasons for them

> [!IMPORTANT]
> The main problem of **`try_as_dyn` and specialization is the compiler's inability, while trait-checking, to distinguish/_discriminate_ between any two given lifetimes**[^1].

[^1]: the compiler cannot _branch_ on whether "`'a : 'b` holds": for soundness, it can either choose not to know the answer, or _assume_ that it holds and produce an obligation for the borrow-checker which shall "assert this" (making compilation fail in a fatal manner if not). Most usages of Rust lie in the latter category (typical `where` clauses anywhere), whilst specialization/`try_as_dyn()` wants to support fallibility of the operation (_i.e._, being queried on a type not fulfilling the predicate without causing a compilation error). This rules out the latter, resulting in the need for the former, _i.e._, for the `try_as_dyn()` attempt to unconditionally "fail" with `None`.

### `'static` is not mentioned anywhere in the `impl` block header.

The most obvious one: if you have `impl IsStatic for &'static str`, then determining whether `&'? str : IsStatic` does hold amounts to discriminating `'? : 'static`.

### Each outlives `where` bound (`Type: 'a` and `'a: 'b`) does not mention lifetime-infected parameters.

Parameters are considered lifetime-infected if they are defined in an `impl` block's generic parameter list.
Const generics are excempt, as they can't affect lifetimes.
`for<'a>` lifetimes (and in the future types) are not lifetime-infected.

We can create lifetime discrimination this way. For instance, given `impl<'a, 'b> Outlives<'a> for &'b str where 'b : 'a {}`, `Outlives<'static>` amounts to `IsStatic` from previous bullet.

### Each lifetime-infected parameter is mentioned at most once in the `Self` type and the implemented trait's generic parameters, combined.

Repetition of a parameter entails equality of those two use-sites; in lifetime-terms, this would be a double `'a : 'b` / `'b : 'a` clause, for instance.
Follow-up from the previous example: `impl<'a> Uses<'a> for &'a str {}`, and check whether `&'? str : Uses<'static>`.

### Each individual trait where bound (`Type: Trait`) mentions each lifetime-infected parameter at most once.

Mentioning a lifetime-infected parameter in multiple `where` bounds is allowed.

Looking at the previous rules, which focuses on `Self : …`, this is just observing that shifting the requirements to other parameters within `where` clauses \[ought to\] boil down to the same set of issues.

This is _unnecessarily restrictive_: we should be able to loosen it up somehow. Repetition only in `where` clauses seems fine.


### The `impl` block is a handwritten impl

as opposed to a type implementing a trait automatically by the compiler (such as auto-traits, `dyn Bounds… : Bounds…`, and closures)


The reason for this is that some such auto-generated impls _come with hidden bounds or whatnot_, which run afoul of the previous rules, whilst also being _extremely challenging for the current compiler logic to know of such bounds_.
IIUC, this restriction could be lifted in the future should the compiler logic be better at spotting these hidden bounds, when present.

One contrived such example being the case of `dyn 'u + for<'a> Outlives<'a>`, where the compiler-generated `impl` for it of `Outlives` is: `impl<'b, 'u> Outlives<'b> for dyn 'u + for<'a> Outlives<'a> where 'b : 'u {}` which violates the "`'a: 'b` not to mention lt-infected params" rule, whilst also being hard to detect in current compiler logic.

### Associated type projections (`<Type as Trait>::Assoc`) are not mentioned anywhere in the `impl` block header.

Associated-type equality bounds can very much amount to lifetime-infected parameter equality constraints,
which are problematic as per the "at most one mention of each lifetime-infected parameter in header" rule.
To illustrate, with the following definitions, `&'? str: Trait<'static>` discriminates `'?` against `'static`:
```rust
trait Trait<'x> {}
impl<'a, 'b> Trait<'b> for &'a str
where
    //     &'a str = &'b str,
    Option<&'a str>: IntoIterator<Item = &'b str>,
{}
```

```rust
trait Trait<'x> {}
impl<'a> Trait<'a> for &'a str {}
```

### Each trait `where` bound with an associated type equality (`Type: Trait<Assoc = Type2>`) does not mention lifetime-infected parameters.

Checking whether `Option<&'? str>: IntoIterator<Item = &'static str>` holds discriminates `'?` against `'static`.
