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

The `span`'s context, given by the method [`ctxt`] and returning [SyntaxContext],
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

If the `m!(x, x.unwrap());` line is expanded, we would get two expanded
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

`Span` provides a method ([`in_external_macro`]) that can
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
if foo_span.in_external_macro(cx.sess().source_map()) {
    // We should ignore macro from a foreign crate.
    return;
}
```

### The `is_from_proc_macro` function
A common point of confusion is the existence of [`is_from_proc_macro`]
and how it differs from the other [`in_external_macro`]/[`from_expansion`] functions.

While [`in_external_macro`] and [`from_expansion`] both work perfectly fine for detecting expanded code
from *declarative* macros (i.e. `macro_rules!` and macros 2.0),
detecting *proc macro*-generated code is a bit more tricky, as proc macros can (and often do)
freely manipulate the span of returned tokens.

In practice, this often happens through the use of [`quote::quote_spanned!`] with a span from the input tokens. 

In those cases, there is no *reliable* way for the compiler (and tools like Clippy)
to distinguish code that comes from such a proc macro from code that the user wrote directly,
and [`in_external_macro`] will return `false`.

This is usually not an issue for the compiler and actually helps proc macro authors create better error messages,
as it allows associating parts of the expansion with parts of the macro input and lets the compiler
point the user to the relevant code in case of a compile error.

However, for Clippy this is inconvenient, because most of the time *we don't* want
to lint proc macro-generated code and this makes it impossible to tell what is and isn't proc macro code.

> NOTE: this is specifically only an issue when a proc macro explicitly sets the span to that of an **input span**.
>
> For example, other common ways of creating `TokenStream`s, such as `"fn foo() {...}".parse::<TokenStream>()`,
> sets each token's span to `Span::call_site()`, which already marks the span as coming from a proc macro
> and the usual span methods have no problem detecting that as a macro span.

As such, Clippy has its own `is_from_proc_macro` function which tries to *approximate*
whether a span comes from a proc macro, by checking whether the source text at the given span
lines up with the given AST node.

This function is typically used in combination with the other mentioned macro span functions,
but is usually called much later into the condition chain as it's a bit heavier than most other conditions,
so that the other cheaper conditions can fail faster. For example, the `borrow_deref_ref` lint:
```rs
impl<'tcx> LateLintPass<'tcx> for BorrowDerefRef {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &rustc_hir::Expr<'tcx>) {
        if let ... = ...
            && ...
            && !e.span.from_expansion()
            && ...
            && ...
            && !is_from_proc_macro(cx, e)
            && ...
        {
            ...
        }
    }
}
```

### Testing lints with macro expansions
To test that all of these cases are handled correctly in your lint,
we have a helper auxiliary crate that exposes various macros, used by tests like so:
```rust
//@aux-build:proc_macros.rs

extern crate proc_macros;

fn main() {
    proc_macros::external!{ code_that_should_trigger_your_lint }
    proc_macros::with_span!{ span code_that_should_trigger_your_lint }
}
```
This exercises two cases:
- `proc_macros::external!` is a simple proc macro that echos the input tokens back but with a macro span:
this represents the usual, common case where an external macro expands to code that your lint would trigger,
and is correctly handled by `in_external_macro` and `Span::from_expansion`.

- `proc_macros::with_span!` echos back the input tokens starting from the second token
with the span of the first token: this is where the other functions will fail and `is_from_proc_macro` is needed


[`ctxt`]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_span/struct.Span.html#method.ctxt
[expansion]: https://rustc-dev-guide.rust-lang.org/macro-expansion.html#expansion-and-ast-integration
[`from_expansion`]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_span/struct.Span.html#method.from_expansion
[`in_external_macro`]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_span/struct.Span.html#method.in_external_macro
[Span]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_span/struct.Span.html
[SyntaxContext]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_span/hygiene/struct.SyntaxContext.html
[`is_from_proc_macro`]: https://doc.rust-lang.org/nightly/nightly-rustc/clippy_utils/fn.is_from_proc_macro.html
[`quote::quote_spanned!`]: https://docs.rs/quote/latest/quote/macro.quote_spanned.html
