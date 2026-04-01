# Emitting a lint

Once we have [defined a lint](defining_lints.md), written [UI
tests](writing_tests.md) and chosen [the lint pass](lint_passes.md) for the lint,
we can begin the implementation of the lint logic so that we can emit it and
gradually work towards a lint that behaves as expected.

Note that we will not go into concrete implementation of a lint logic in this
chapter. We will go into details in later chapters as well as in two examples of
real Clippy lints.

To emit a lint, we must implement a pass (see [Lint Passes](lint_passes.md)) for
the lint that we have declared. In this example we'll implement a "late" lint,
so take a look at the [LateLintPass][late_lint_pass] documentation, which
provides an abundance of methods that we can implement for our lint.

```rust
pub trait LateLintPass<'tcx>: LintPass {
    // Trait methods
}
```

By far the most common method used for Clippy lints is [`check_expr`
method][late_check_expr], this is because Rust is an expression language and,
more often than not, the lint we want to work on must examine expressions.

> _Note:_ If you don't fully understand what expressions are in Rust, take a
> look at the official documentation on [expressions][rust_expressions]

Other common ones include the [`check_fn` method][late_check_fn] and the
[`check_item` method][late_check_item].

### Emitting a lint

Inside the trait method that we implement, we can write down the lint logic and
emit the lint with suggestions.

Clippy's [diagnostics] provides quite a few diagnostic functions that we can use
to emit lints. Take a look at the documentation to pick one that suits your
lint's needs the best. Some common ones you will encounter in the Clippy
repository includes:

- [`span_lint`]: Emits a lint without providing any other information
- [`span_lint_and_note`]: Emits a lint and adds a note
- [`span_lint_and_help`]: Emits a lint and provides a helpful message
- [`span_lint_and_sugg`]: Emits a lint and provides a suggestion to fix the code
- [`span_lint_and_then`]: Like `span_lint`, but allows for a lot of output
  customization.

```rust
impl<'tcx> LateLintPass<'tcx> for LintName {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>)  {
        // Imagine that `some_lint_expr_logic` checks for requirements for emitting the lint
        if some_lint_expr_logic(expr) {
            span_lint_and_help(
                cx, // < The context
                LINT_NAME, // < The name of the lint in ALL CAPS
                expr.span, // < The span to lint
                "message on why the lint is emitted",
                None, // < An optional help span (to highlight something in the lint)
                "message that provides a helpful suggestion",
            );
        }
    }
}
```

> Note: The message should be matter of fact and avoid capitalization and
> punctuation. If multiple sentences are needed, the messages should probably be
> split up into an error + a help / note / suggestion message.

## Suggestions: Automatic fixes

Some lints know what to change in order to fix the code. For example, the lint
[`range_plus_one`][range_plus_one] warns for ranges where the user wrote `x..y +
1` instead of using an [inclusive range][inclusive_range] (`x..=y`). The fix to
this code would be changing the `x..y + 1` expression to `x..=y`. **This is
where suggestions come in**.

A suggestion is a change that the lint provides to fix the issue it is linting.
The output looks something like this (from the example earlier):

```text
error: an inclusive range would be more readable
  --> tests/ui/range_plus_minus_one.rs:37:14
   |
LL |     for _ in 1..1 + 1 {}
   |              ^^^^^^^^ help: use: `1..=1`
```

**Not all suggestions are always right**, some of them require human
supervision, that's why we have [Applicability][applicability].

Applicability indicates confidence in the correctness of the suggestion, some
are always right (`Applicability::MachineApplicable`), but we use
`Applicability::MaybeIncorrect` and others when talking about a suggestion that
may be incorrect.

### Example

The same lint `LINT_NAME` but that emits a suggestion would look something like this:

```rust
impl<'tcx> LateLintPass<'tcx> for LintName {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>)  {
        // Imagine that `some_lint_expr_logic` checks for requirements for emitting the lint
        if some_lint_expr_logic(expr) {
            span_lint_and_sugg( // < Note this change
                cx,
                LINT_NAME,
                span,
                "message on why the lint is emitted",
                "use",
                format!("foo + {} * bar", snippet(cx, expr.span, "<default>")), // < Suggestion
                Applicability::MachineApplicable,
            );
        }
    }
}
```

Suggestions generally use the [`format!`][format_macro] macro to interpolate the
old values with the new ones. To get code snippets, use one of the `snippet*`
functions from `clippy_utils::source`.

## How to choose between notes, help messages and suggestions

Notes are presented separately from the main lint message, they provide useful
information that the user needs to understand why the lint was activated. They
are the most helpful when attached to a span.

Examples:

### Notes

```text
error: calls to `std::mem::forget` with a reference instead of an owned value. Forgetting a reference does nothing.
  --> tests/ui/drop_forget_ref.rs:10:5
   |
10 |     forget(&SomeStruct);
   |     ^^^^^^^^^^^^^^^^^^^
   |
   = note: `-D clippy::forget-ref` implied by `-D warnings`
note: argument has type &SomeStruct
  --> tests/ui/drop_forget_ref.rs:10:12
   |
10 |     forget(&SomeStruct);
   |            ^^^^^^^^^^^
```

### Help Messages

Help messages are specifically to help the user. These are used in situation
where you can't provide a specific machine applicable suggestion. They can also
be attached to a span.

Example:

```text
error: constant division of 0.0 with 0.0 will always result in NaN
  --> tests/ui/zero_div_zero.rs:6:25
   |
6  |     let other_f64_nan = 0.0f64 / 0.0;
   |                         ^^^^^^^^^^^^
   |
   = help: consider using `f64::NAN` if you would like a constant representing NaN
```

### Suggestions

Suggestions are the most helpful, they are changes to the source code to fix the
error. The magic in suggestions is that tools like `rustfix` can detect them and
automatically fix your code.

Example:

```text
error: This `.fold` can be more succinctly expressed as `.any`
--> tests/ui/methods.rs:390:13
    |
390 |     let _ = (0..3).fold(false, |acc, x| acc || x > 2);
    |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `.any(|x| x > 2)`
    |
```

### Snippets

Snippets are pieces of the source code (as a string), they are extracted
generally using the [`snippet`][snippet_fn] function.

For example, if you want to know how an item looks (and you know the item's
span), you could use `snippet(cx, span, "..")`.

## Final: Run UI Tests to Emit the Lint

Now, if we run our [UI test](writing_tests.md), we should see that Clippy now
produces output that contains the lint message we designed.

The next step is to implement the logic properly, which is a detail that we will
cover in the next chapters.

[diagnostics]: https://doc.rust-lang.org/nightly/nightly-rustc/clippy_utils/diagnostics/index.html
[late_check_expr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/trait.LateLintPass.html#method.check_expr
[late_check_fn]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/trait.LateLintPass.html#method.check_fn
[late_check_item]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/trait.LateLintPass.html#method.check_item
[late_lint_pass]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/trait.LateLintPass.html
[rust_expressions]: https://doc.rust-lang.org/reference/expressions.html
[`span_lint`]: https://doc.rust-lang.org/beta/nightly-rustc/clippy_utils/diagnostics/fn.span_lint.html
[`span_lint_and_note`]: https://doc.rust-lang.org/beta/nightly-rustc/clippy_utils/diagnostics/fn.span_lint_and_note.html
[`span_lint_and_help`]: https://doc.rust-lang.org/nightly/nightly-rustc/clippy_utils/diagnostics/fn.span_lint_and_help.html
[`span_lint_and_sugg`]: https://doc.rust-lang.org/nightly/nightly-rustc/clippy_utils/diagnostics/fn.span_lint_and_sugg.html
[`span_lint_and_then`]: https://doc.rust-lang.org/beta/nightly-rustc/clippy_utils/diagnostics/fn.span_lint_and_then.html
[range_plus_one]: https://rust-lang.github.io/rust-clippy/master/index.html#range_plus_one
[inclusive_range]: https://doc.rust-lang.org/std/ops/struct.RangeInclusive.html
[applicability]: https://doc.rust-lang.org/beta/nightly-rustc/rustc_errors/enum.Applicability.html
[snippet_fn]: https://doc.rust-lang.org/beta/nightly-rustc/clippy_utils/source/fn.snippet.html
[format_macro]: https://doc.rust-lang.org/std/macro.format.html
