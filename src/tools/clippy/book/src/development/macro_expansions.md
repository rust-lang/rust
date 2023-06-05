# Dealing with macros and expansions

Sometimes we might encounter Rust macro expansions while working with Clippy.
While macro expansions are not as dramatic and profound as the expansion
of our universe, they can certainly bring chaos to the orderly world
of code and logic.

The general rule of thumb is that we should ignore code with macro
expansions when working with Clippy because the code can be dynamic
in ways that are difficult or impossible for us to foresee.

## False Positives

What exactly do we mean by _dynamic in ways that are difficult to foresee_?

Macros are [expanded][expansion] in the `EarlyLintPass` level,
so the Abstract Syntax Tree (AST) is generated in place of macros.
This means the code which we work with in Clippy is already expanded.

If we wrote a new lint, there is a possibility that the lint is
triggered in macro-generated code. Since this expanded macro code
is not written by the macro's user but really by the macro's author,
the user cannot and should not be responsible for fixing the issue
that triggers the lint.

Besides, a [Span] in a macro can be changed by the macro author.
Therefore, any lint check related to lines or columns should be
avoided since they might be changed at any time and become unreliable
or incorrect information.

Because of these unforeseeable or unstable behaviors, macro expansion
should often not be regarded as a part of the stable API.
This is also why most lints check if they are inside a macro or not
before emitting suggestions to the end user to avoid false positives.

## How to Work with Macros

Several functions are available for working with macros.

### The `Span.from_expansion` method

We could utilize a `span`'s [`from_expansion`] method, which
detects if the `span` is from a macro expansion / desugaring.
This is a very common first step in a lint:

```rust
if expr.span.from_expansion() {
    // We most likely want to ignore it.
    return;
}
```

### `Span.ctxt` method

The `span`'s context, given by the method [`ctxt`] and returning [SpanContext],
represents if the span is from a macro expansion and, if it is, which
macro call expanded this span.

Sometimes, it is useful to check if the context of two spans are equal.
For instance, suppose we have the following line of code that would
expand into `1 + 0`:

```rust
// The following code expands to `1 + 0` for both `EarlyLintPass` and `LateLintPass`
1 + mac!()
```

Assuming that we'd collect the `1` expression as a variable `left` and the
`0`/`mac!()` expression as a variable `right`, we can simply compare their
contexts. If the context is different, then we most likely are dealing with a
macro expansion and should just ignore it:

```rust
if left.span.ctxt() != right.span.ctxt() {
    // The code author most likely cannot modify this expression
    return;
}
```

> **Note**: Code that is not from expansion is in the "root" context.
> So any spans whose `from_expansion` returns `false` can be assumed
> to have the same context. Because of this, using `span.from_expansion()`
> is often sufficient.

Going a bit deeper, in a simple expression such as `a == b`,
`a` and `b` have the same context.
However, in a `macro_rules!` with `a == $b`, `$b` is expanded to
an expression that contains a different context from `a`.

Take a look at the following macro `m`:

```rust
macro_rules! m {
    ($a:expr, $b:expr) => {
        if $a.is_some() {
            $b;
        }
    }
}

let x: Option<u32> = Some(42);
m!(x, x.unwrap());
```

If the `m!(x, x.unwrapp());` line is expanded, we would get two expanded
expressions:

- `x.is_some()` (from the `$a.is_some()` line in the `m` macro)
- `x.unwrap()` (corresponding to `$b` in the `m` macro)

Suppose `x.is_some()` expression's span is associated with the `x_is_some_span` variable
and `x.unwrap()` expression's span is associated with `x_unwrap_span` variable,
we could assume that these two spans do not share the same context:

```rust
// x.is_some() is from inside the macro
// x.unwrap() is from outside the macro
assert_ne!(x_is_some_span.ctxt(), x_unwrap_span.ctxt());
```

### The `in_external_macro` function

`rustc_middle::lint` provides a function ([`in_external_macro`]) that can
detect if the given span is from a macro defined in a foreign crate.

Therefore, if we really want a new lint to work with macro-generated code,
this is the next line of defense to avoid macros not defined inside
the current crate since it is unfair to the user if Clippy lints code
which the user cannot change.

For example, assume we have the following code that is being examined
by Clippy:

```rust
#[macro_use]
extern crate a_foreign_crate_with_macros;

// `foo` macro is defined in `a_foreign_crate_with_macros`
foo!("bar");
```

Also assume that we get the corresponding variable `foo_span` for the
`foo` macro call, we could decide not to lint if `in_external_macro`
results in `true` (note that `cx` can be `EarlyContext` or `LateContext`):

```rust
if in_external_macro(cx.sess(), foo_span) {
    // We should ignore macro from a foreign crate.
    return;
}
```

[`ctxt`]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_span/struct.Span.html#method.ctxt
[expansion]: https://rustc-dev-guide.rust-lang.org/macro-expansion.html#expansion-and-ast-integration
[`from_expansion`]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_span/struct.Span.html#method.from_expansion
[`in_external_macro`]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_middle/lint/fn.in_external_macro.html
[Span]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_span/struct.Span.html
[SpanContext]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_span/hygiene/struct.SyntaxContext.html
