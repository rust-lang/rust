# Lint passes

Before working on the logic of a new lint, there is an important decision
that every Clippy developer must make: to use
[`EarlyLintPass`][early_lint_pass] or [`LateLintPass`][late_lint_pass].

In short, the `LateLintPass` has access to type and symbol information while the
`EarlyLintPass` doesn't. If you don't need access to type information, use the
`EarlyLintPass`.

Let us expand on these two traits more below.

## `EarlyLintPass`

If you examine the documentation on [`EarlyLintPass`][early_lint_pass] closely,
you'll see that every method defined for this trait utilizes a
[`EarlyContext`][early_context]. In `EarlyContext`'s documentation, it states:

> Context for lint checking of the AST, after expansion, before lowering to HIR.

Voilà. `EarlyLintPass` works only on the Abstract Syntax Tree (AST) level.
And AST is generated during the [lexing and parsing][lexing_and_parsing] phase
of code compilation. Therefore, it doesn't know what a symbol means or information about types, and it should
be our trait choice for a new lint if the lint only deals with syntax-related issues.

While linting speed has not been a concern for Clippy,
the `EarlyLintPass` is faster, and it should be your choice
if you know for sure a lint does not need type information.

As a reminder, run the following command to generate boilerplate for lints
that use `EarlyLintPass`:

```sh
$ cargo dev new_lint --name=<your_new_lint> --pass=early --category=<your_category_choice>
```

### Example for `EarlyLintPass`

Take a look at the following code:

```rust
let x = OurUndefinedType;
x.non_existing_method();
```

From the AST perspective, both lines are "grammatically" correct.
The assignment uses a `let` and ends with a semicolon. The invocation
of a method looks fine, too. As programmers, we might raise a few
questions already, but the parser is okay with it. This is what we
mean when we say `EarlyLintPass` deals with only syntax on the AST level.

Alternatively, think of the `foo_functions` lint we mentioned in
the [Define New Lints](defining_lints.md) chapter.

We want the `foo_functions` lint to detect functions with `foo` as their name.
Writing a lint that only checks for the name of a function means that we only
work with the AST and don't have to access the type system at all (the type system is where
`LateLintPass` comes into the picture).

## `LateLintPass`

In contrast to `EarlyLintPass`, `LateLintPass` contains type information.

If you examine the documentation on [`LateLintPass`][late_lint_pass] closely,
you see that every method defined in this trait utilizes a
[`LateContext`][late_context].

In `LateContext`'s documentation we will find methods that
deal with type-checking, which do not exist in `EarlyContext`, such as:

- [`maybe_typeck_results`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/context/struct.LateContext.html#method.maybe_typeck_results)
- [`typeck_results`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/context/struct.LateContext.html#method.typeck_results)

### Example for `LateLintPass`

Let us take a look with the following example:

```rust
let x = OurUndefinedType;
x.non_existing_method();
```

These two lines of code are syntactically correct code from the perspective
of the AST. We have an assignment and invoke a method on the variable that
is of a type. Grammatically, everything is in order for the parser.

However, going down a level and looking at the type information,
the compiler will notice that both `OurUndefinedType` and `non_existing_method()`
**are undefined**.

As Clippy developers, to access such type information, we must implement
`LateLintPass` on our lint.
When you browse through Clippy's lints, you will notice that almost every lint
is implemented in a `LateLintPass`, specifically because we often need to check
not only for syntactic issues but also type information.

Another limitation of the `EarlyLintPass` is that the nodes are only identified
by their position in the AST. This means that you can't just get an `id` and
request a certain node. For most lints that is fine, but we have some lints
that require the inspection of other nodes, which is easier at the HIR level.
In these cases, `LateLintPass` is the better choice.

As a reminder, run the following command to generate boilerplate for lints
that use `LateLintPass`:

```sh
$ cargo dev new_lint --name=<your_new_lint> --pass=late --category=<your_category_choice>
```

[early_context]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/context/struct.EarlyContext.html
[early_lint_pass]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/trait.EarlyLintPass.html
[late_context]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/context/struct.LateContext.html
[late_lint_pass]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/trait.LateLintPass.html
[lexing_and_parsing]: https://rustc-dev-guide.rust-lang.org/overview.html#lexing-and-parsing
