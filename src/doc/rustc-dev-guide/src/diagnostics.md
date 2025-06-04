# Errors and lints

<!-- toc -->

A lot of effort has been put into making `rustc` have great error messages.
This chapter is about how to emit compile errors and lints from the compiler.

## Diagnostic structure

The main parts of a diagnostic error are the following:

```
error[E0000]: main error message
  --> file.rs:LL:CC
   |
LL | <code>
   | -^^^^- secondary label
   |  |
   |  primary label
   |
   = note: note without a `Span`, created with `.note`
note: sub-diagnostic message for `.span_note`
  --> file.rs:LL:CC
   |
LL | more code
   |      ^^^^
```

- Level (`error`, `warning`, etc.). It indicates the severity of the message.
  (See [diagnostic levels](#diagnostic-levels))
- Code (for example, for "mismatched types", it is `E0308`). It helps
  users get more information about the current error through an extended
  description of the problem in the error code index. Not all diagnostic have a
  code. For example, diagnostics created by lints don't have one.
- Message. It is the main description of the problem. It should be general and
  able to stand on its own, so that it can make sense even in isolation.
- Diagnostic window. This contains several things:
  - The path, line number and column of the beginning of the primary span.
  - The users' affected code and its surroundings.
  - Primary and secondary spans underlying the users' code. These spans can
    optionally contain one or more labels.
    - Primary spans should have enough text to describe the problem in such a
      way that if it were the only thing being displayed (for example, in an
      IDE) it would still make sense. Because it is "spatially aware" (it
      points at the code), it can generally be more succinct than the error
      message.
    - If cluttered output can be foreseen in cases when multiple span labels
      overlap, it is a good idea to tweak the output appropriately. For
      example, the `if/else arms have incompatible types` error uses different
      spans depending on whether the arms are all in the same line, if one of
      the arms is empty and if none of those cases applies.
- Sub-diagnostics. Any error can have multiple sub-diagnostics that look
  similar to the main part of the error. These are used for cases where the
  order of the explanation might not correspond with the order of the code. If
  the order of the explanation can be "order free", leveraging secondary labels
  in the main diagnostic is preferred, as it is typically less verbose.

The text should be matter of fact and avoid capitalization and periods, unless
multiple sentences are _needed_:

```txt
error: the fobrulator needs to be krontrificated
```

When code or an identifier must appear in a message or label, it should be
surrounded with backticks:

```txt
error: the identifier `foo.bar` is invalid
```

### Error codes and explanations

Most errors have an associated error code. Error codes are linked to long-form
explanations which contains an example of how to trigger the error and in-depth
details about the error. They may be viewed with the `--explain` flag, or via
the [error index].

As a general rule, give an error a code (with an associated explanation) if the
explanation would give more information than the error itself. A lot of the time
it's better to put all the information in the emitted error itself. However,
sometimes that would make the error verbose or there are too many possible
triggers to include useful information for all cases in the error, in which case
it's a good idea to add an explanation.[^estebank]
As always, if you are not sure, just ask your reviewer!

If you decide to add a new error with an associated error code, please read
[this section][error-codes] for a guide and important details about the
process.

[^estebank]: This rule of thumb was suggested by **@estebank** [here][estebank-comment].

[error index]: https://doc.rust-lang.org/error-index.html
[estebank-comment]: https://github.com/rust-lang/rustc-dev-guide/pull/967#issuecomment-733218283
[error-codes]: ./diagnostics/error-codes.md

### Lints versus fixed diagnostics

Some messages are emitted via [lints](#lints), where the user can control the
level. Most diagnostics are hard-coded such that the user cannot control the
level.

Usually it is obvious whether a diagnostic should be "fixed" or a lint, but
there are some grey areas.

Here are a few examples:

- Borrow checker errors: these are fixed errors. The user cannot adjust the
  level of these diagnostics to silence the borrow checker.
- Dead code: this is a lint. While the user probably doesn't want dead code in
  their crate, making this a hard error would make refactoring and development
  very painful.
- [future-incompatible lints]:
  these are silenceable lints.
  It was decided that making them fixed errors would cause too much breakage,
  so warnings are instead emitted,
  and will eventually be turned into fixed (hard) errors.

Hard-coded warnings (those using methods like `span_warn`) should be avoided
for normal code, preferring to use lints instead. Some cases, such as warnings
with CLI flags, will require the use of hard-coded warnings.

See the `deny` [lint level](#diagnostic-levels) below for guidelines when to
use an error-level lint instead of a fixed error.

[future-incompatible lints]: #future-incompatible-lints

## Diagnostic output style guide

- Write in plain simple English. If your message, when shown on a – possibly
  small – screen (which hasn't been cleaned for a while), cannot be understood
  by a normal programmer, who just came out of bed after a night partying,
  it's too complex.
- `Error`, `Warning`, `Note`, and `Help` messages start with a lowercase
  letter and do not end with punctuation.
- Error messages should be succinct. Users will see these error messages many
  times, and more verbose descriptions can be viewed with the `--explain`
  flag. That said, don't make it so terse that it's hard to understand.
- The word "illegal" is illegal. Prefer "invalid" or a more specific word
  instead.
- Errors should document the span of code where they occur (use
  [`rustc_errors::DiagCtxt`][DiagCtxt]'s
  `span_*` methods or a diagnostic struct's `#[primary_span]` to easily do
  this). Also `note` other spans that have contributed to the error if the span
  isn't too large.
- When emitting a message with span, try to reduce the span to the smallest
  amount possible that still signifies the issue
- Try not to emit multiple error messages for the same error. This may require
  detecting duplicates.
- When the compiler has too little information for a specific error message,
  consult with the compiler team to add new attributes for library code that
  allow adding more information. For example see
  [`#[rustc_on_unimplemented]`](#rustc_on_unimplemented). Use these
  annotations when available!
- Keep in mind that Rust's learning curve is rather steep, and that the
  compiler messages are an important learning tool.
- When talking about the compiler, call it `the compiler`, not `Rust` or
  `rustc`.
- Use the [Oxford comma](https://en.wikipedia.org/wiki/Serial_comma) when
  writing lists of items.

### Lint naming

From [RFC 0344], lint names should be consistent, with the following
guidelines:

The basic rule is: the lint name should make sense when read as "allow
*lint-name*" or "allow *lint-name* items". For example, "allow
`deprecated` items" and "allow `dead_code`" makes sense, while "allow
`unsafe_block`" is ungrammatical (should be plural).

- Lint names should state the bad thing being checked for, e.g. `deprecated`,
  so that `#[allow(deprecated)]` (items) reads correctly. Thus `ctypes` is not
  an appropriate name; `improper_ctypes` is.

- Lints that apply to arbitrary items (like the stability lints) should just
  mention what they check for: use `deprecated` rather than
  `deprecated_items`. This keeps lint names short. (Again, think "allow
  *lint-name* items".)

- If a lint applies to a specific grammatical class, mention that class and
  use the plural form: use `unused_variables` rather than `unused_variable`.
  This makes `#[allow(unused_variables)]` read correctly.

- Lints that catch unnecessary, unused, or useless aspects of code should use
  the term `unused`, e.g. `unused_imports`, `unused_typecasts`.

- Use snake case in the same way you would for function names.

[RFC 0344]: https://github.com/rust-lang/rfcs/blob/master/text/0344-conventions-galore.md#lints

### Diagnostic levels

Guidelines for different diagnostic levels:

- `error`: emitted when the compiler detects a problem that makes it unable to
  compile the program, either because the program is invalid or the programmer
  has decided to make a specific `warning` into an error.

- `warning`: emitted when the compiler detects something odd about a program.
  Care should be taken when adding warnings to avoid warning fatigue, and
  avoid false-positives where there really isn't a problem with the code. Some
  examples of when it is appropriate to issue a warning:

  - A situation where the user *should* take action, such as swap out a
    deprecated item, or use a `Result`, but otherwise doesn't prevent
    compilation.
  - Unnecessary syntax that can be removed without affecting the semantics of
    the code. For example, unused code, or unnecessary `unsafe`.
  - Code that is very likely to be incorrect, dangerous, or confusing, but the
    language technically allows, and is not ready or confident enough to make
    an error. For example `unused_comparisons` (out of bounds comparisons) or
    `bindings_with_variant_name` (the user likely did not intend to create a
    binding in a pattern).
  - [Future-incompatible lints](#future-incompatible), where something was
    accidentally or erroneously accepted in the past, but rejecting would
    cause excessive breakage in the ecosystem.
  - Stylistic choices. For example, camel or snake case, or the `dyn` trait
    warning in the 2018 edition. These have a high bar to be added, and should
    only be used in exceptional circumstances. Other stylistic choices should
    either be allow-by-default lints, or part of other tools like Clippy or
    rustfmt.

- `help`: emitted following an `error` or `warning` to give additional
  information to the user about how to solve their problem. These messages
  often include a suggestion string and [`rustc_errors::Applicability`]
  confidence level to guide automated source fixes by tools. See the
  [Suggestions](#suggestions) section for more details.

  The error or warning portion should *not* suggest how to fix the problem,
  only the "help" sub-diagnostic should.

- `note`: emitted to given more context and identify additional circumstances
  and parts of the code that caused the warning or error. For example, the
  borrow checker will note any previous conflicting borrows.

  `help` vs `note`: `help` should be used to show changes the user can
  possibly make to fix the problem. `note` should be used for everything else,
  such as other context, information and facts, online resources to read, etc.

Not to be confused with *lint levels*, whose guidelines are:

- `forbid`: Lints should never default to `forbid`.
- `deny`: Equivalent to `error` diagnostic level. Some examples:

  - A future-incompatible or edition-based lint that has graduated from the
    warning level.
  - Something that has an extremely high confidence that is incorrect, but
    still want an escape hatch to allow it to pass.

- `warn`: Equivalent to the `warning` diagnostic level. See `warning` above
  for guidelines.
- `allow`: Examples of the kinds of lints that should default to `allow`:

  - The lint has a too high false positive rate.
  - The lint is too opinionated.
  - The lint is experimental.
  - The lint is used for enforcing something that is not normally enforced.
    For example, the `unsafe_code` lint can be used to prevent usage of unsafe
    code.

More information about lint levels can be found in the [rustc
book][rustc-lint-levels] and the [reference][reference-diagnostics].

[`rustc_errors::Applicability`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/enum.Applicability.html
[reference-diagnostics]: https://doc.rust-lang.org/nightly/reference/attributes/diagnostics.html#lint-check-attributes
[rustc-lint-levels]: https://doc.rust-lang.org/nightly/rustc/lints/levels.html

## Helpful tips and options

### Finding the source of errors

There are three main ways to find where a given error is emitted:

- `grep` for either a sub-part of the error message/label or error code. This
  usually works well and is straightforward, but there are some cases where
  the code emitting the error is removed from the code where the error is
  constructed behind a relatively deep call-stack. Even then, it is a good way
  to get your bearings.
- Invoking `rustc` with the nightly-only flag `-Z treat-err-as-bug=1`
  will treat the first error being emitted as an Internal Compiler Error, which
  allows you to get a
  stack trace at the point the error has been emitted. Change the `1` to
  something else if you wish to trigger on a later error.

  There are limitations with this approach:
  - Some calls get elided from the stack trace because they get inlined in the compiled `rustc`.
  - The _construction_ of the error is far away from where it is _emitted_,
    a problem similar to the one we faced with the `grep` approach.
    In some cases, we buffer multiple errors in order to emit them in order.
- Invoking `rustc` with `-Z track-diagnostics` will print error creation
  locations alongside the error.

The regular development practices apply: judicious use of `debug!()` statements
and use of a debugger to trigger break points in order to figure out in what
order things are happening.

## `Span`

[`Span`][span] is the primary data structure in `rustc` used to represent a
location in the code being compiled. `Span`s are attached to most constructs in
HIR and MIR, allowing for more informative error reporting.

[span]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/struct.Span.html

A `Span` can be looked up in a [`SourceMap`][sourcemap] to get a "snippet"
useful for displaying errors with [`span_to_snippet`][sptosnip] and other
similar methods on the `SourceMap`.

[sourcemap]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/source_map/struct.SourceMap.html
[sptosnip]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/source_map/struct.SourceMap.html#method.span_to_snippet

## Error messages

The [`rustc_errors`][errors] crate defines most of the utilities used for
reporting errors.

[errors]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/index.html

Diagnostics can be implemented as types which implement the `Diagnostic`
trait. This is preferred for new diagnostics as it enforces a separation
between diagnostic emitting logic and the main code paths. For less-complex
diagnostics, the `Diagnostic` trait can be derived -- see [Diagnostic
structs][diagnostic-structs]. Within the trait implementation, the APIs
described below can be used as normal.

[diagnostic-structs]: ./diagnostics/diagnostic-structs.md

[`DiagCtxt`][DiagCtxt] has methods that create and emit errors. These methods
usually have names like `span_err` or `struct_span_err` or `span_warn`, etc...
There are lots of them; they emit different types of "errors", such as
warnings, errors, fatal errors, suggestions, etc.

[DiagCtxt]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.DiagCtxt.html

In general, there are two classes of such methods: ones that emit an error
directly and ones that allow finer control over what to emit. For example,
[`span_err`][spanerr] emits the given error message at the given `Span`, but
[`struct_span_err`][strspanerr] instead returns a
[`Diag`][diag].

Most of these methods will accept strings, but it is recommended that typed
identifiers for translatable diagnostics be used for new diagnostics (see
[Translation][translation]).

[translation]: ./diagnostics/translation.md

`Diag` allows you to add related notes and suggestions to an error
before emitting it by calling the [`emit`][emit] method. (Failing to either
emit or [cancel][cancel] a `Diag` will result in an ICE.) See the
[docs][diag] for more info on what you can do.

[spanerr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.DiagCtxt.html#method.span_err
[strspanerr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.DiagCtxt.html#method.struct_span_err
[diag]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.Diag.html
[emit]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.Diag.html#method.emit
[cancel]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.Diag.html#method.cancel

```rust,ignore
// Get a `Diag`. This does _not_ emit an error yet.
let mut err = sess.dcx.struct_span_err(sp, fluent::example::example_error);

// In some cases, you might need to check if `sp` is generated by a macro to
// avoid printing weird errors about macro-generated code.

if let Ok(snippet) = sess.source_map().span_to_snippet(sp) {
    // Use the snippet to generate a suggested fix
    err.span_suggestion(suggestion_sp, fluent::example::try_qux_suggestion, format!("qux {}", snippet));
} else {
    // If we weren't able to generate a snippet, then emit a "help" message
    // instead of a concrete "suggestion". In practice this is unlikely to be
    // reached.
    err.span_help(suggestion_sp, fluent::example::qux_suggestion);
}

// emit the error
err.emit();
```

```fluent
example-example-error = oh no! this is an error!
  .try-qux-suggestion = try using a qux here
  .qux-suggestion = you could use a qux here instead
```

## Suggestions

In addition to telling the user exactly _why_ their code is wrong, it's
oftentimes furthermore possible to tell them how to fix it. To this end,
[`Diag`][diag] offers a structured suggestions API, which formats code
suggestions pleasingly in the terminal, or (when the `--error-format json` flag
is passed) as JSON for consumption by tools like [`rustfix`][rustfix].

[diag]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.Diag.html
[rustfix]: https://github.com/rust-lang/rustfix

Not all suggestions should be applied mechanically, they have a degree of
confidence in the suggested code, from high
(`Applicability::MachineApplicable`) to low (`Applicability::MaybeIncorrect`).
Be conservative when choosing the level. Use the
[`span_suggestion`][span_suggestion] method of `Diag` to
make a suggestion. The last argument provides a hint to tools whether
the suggestion is mechanically applicable or not.

Suggestions point to one or more spans with corresponding code that will
replace their current content.

The message that accompanies them should be understandable in the following
contexts:

- shown as an independent sub-diagnostic (this is the default output)
- shown as a label pointing at the affected span (this is done automatically if
some heuristics for verbosity are met)
- shown as a `help` sub-diagnostic with no content (used for cases where the
suggestion is obvious from the text, but we still want to let tools to apply
them)
- not shown (used for _very_ obvious cases, but we still want to allow tools to
apply them)

[span_suggestion]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.Diag.html#method.span_suggestion

For example, to make our `qux` suggestion machine-applicable, we would do:

```rust,ignore
let mut err = sess.dcx.struct_span_err(sp, fluent::example::message);

if let Ok(snippet) = sess.source_map().span_to_snippet(sp) {
    err.span_suggestion(
        suggestion_sp,
        fluent::example::try_qux_suggestion,
        format!("qux {}", snippet),
        Applicability::MachineApplicable,
    );
} else {
    err.span_help(suggestion_sp, fluent::example::qux_suggestion);
}

err.emit();
```

This might emit an error like

```console
$ rustc mycode.rs
error[E0999]: oh no! this is an error!
 --> mycode.rs:3:5
  |
3 |     sad()
  |     ^ help: try using a qux here: `qux sad()`

error: aborting due to previous error

For more information about this error, try `rustc --explain E0999`.
```

In some cases, like when the suggestion spans multiple lines or when there are
multiple suggestions, the suggestions are displayed on their own:

```console
error[E0999]: oh no! this is an error!
 --> mycode.rs:3:5
  |
3 |     sad()
  |     ^
help: try using a qux here:
  |
3 |     qux sad()
  |     ^^^

error: aborting due to previous error

For more information about this error, try `rustc --explain E0999`.
```

The possible values of [`Applicability`][appl] are:

- `MachineApplicable`: Can be applied mechanically.
- `HasPlaceholders`: Cannot be applied mechanically because it has placeholder
  text in the suggestions. For example: ```try adding a type: `let x:
  <type>` ```.
- `MaybeIncorrect`: Cannot be applied mechanically because the suggestion may
  or may not be a good one.
- `Unspecified`: Cannot be applied mechanically because we don't know which
  of the above cases it falls into.

[appl]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/enum.Applicability.html

### Suggestion Style Guide

- Suggestions should not be a question. In particular, language like "did you
  mean" should be avoided. Sometimes, it's unclear why a particular suggestion
  is being made. In these cases, it's better to be upfront about what the
  suggestion is.

  Compare "did you mean: `Foo`" vs. "there is a struct with a similar name: `Foo`".

- The message should not contain any phrases like "the following", "as shown",
  etc. Use the span to convey what is being talked about.
- The message may contain further instruction such as "to do xyz, use" or "to do
  xyz, use abc".
- The message may contain a name of a function, variable, or type, but avoid
  whole expressions.

## Lints

The compiler linting infrastructure is defined in the [`rustc_middle::lint`][rlint]
module.

[rlint]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/lint/index.html

### When do lints run?

Different lints will run at different times based on what information the lint
needs to do its job. Some lints get grouped into *passes* where the lints
within a pass are processed together via a single visitor. Some of the passes
are:

- Pre-expansion pass: Works on [AST nodes] before [macro expansion]. This
  should generally be avoided.
  - Example: [`keyword_idents`] checks for identifiers that will become
    keywords in future editions, but is sensitive to identifiers used in
    macros.

- Early lint pass: Works on [AST nodes] after [macro expansion] and name
  resolution, just before [AST lowering]. These lints are for purely
  syntactical lints.
  - Example: The [`unused_parens`] lint checks for parenthesized-expressions
    in situations where they are not needed, like an `if` condition.

- Late lint pass: Works on [HIR nodes], towards the end of [analysis] (after
  borrow checking, etc.). These lints have full type information available.
  Most lints are late.
  - Example: The [`invalid_value`] lint (which checks for obviously invalid
    uninitialized values) is a late lint because it needs type information to
    figure out whether a type allows being left uninitialized.

- MIR pass: Works on [MIR nodes]. This isn't quite the same as other passes;
  lints that work on MIR nodes have their own methods for running.
  - Example: The [`arithmetic_overflow`] lint is emitted when it detects a
    constant value that may overflow.

Most lints work well via the pass systems, and they have a fairly
straightforward interface and easy way to integrate (mostly just implementing
a specific `check` function). However, some lints are easier to write when
they live on a specific code path anywhere in the compiler. For example, the
[`unused_mut`] lint is implemented in the borrow checker as it requires some
information and state in the borrow checker.

Some of these inline lints fire before the linting system is ready. Those
lints will be *buffered* where they are held until later phases of the
compiler when the linting system is ready. See [Linting early in the
compiler](#linting-early-in-the-compiler).


[AST nodes]: the-parser.md
[AST lowering]: ast-lowering.md
[HIR nodes]: hir.md
[MIR nodes]: mir/index.md
[macro expansion]: macro-expansion.md
[analysis]: part-4-intro.md
[`keyword_idents`]: https://doc.rust-lang.org/rustc/lints/listing/allowed-by-default.html#keyword-idents
[`unused_parens`]: https://doc.rust-lang.org/rustc/lints/listing/warn-by-default.html#unused-parens
[`invalid_value`]: https://doc.rust-lang.org/rustc/lints/listing/warn-by-default.html#invalid-value
[`arithmetic_overflow`]: https://doc.rust-lang.org/rustc/lints/listing/deny-by-default.html#arithmetic-overflow
[`unused_mut`]: https://doc.rust-lang.org/rustc/lints/listing/warn-by-default.html#unused-mut

### Lint definition terms

Lints are managed via the [`LintStore`][LintStore] and get registered in
various ways. The following terms refer to the different classes of lints
generally based on how they are registered.

- *Built-in* lints are defined inside the compiler source.
- *Driver-registered* lints are registered when the compiler driver is created
  by an external driver. This is the mechanism used by Clippy, for example.
- *Tool* lints are lints with a path prefix like `clippy::` or `rustdoc::`.
- *Internal* lints are the `rustc::` scoped tool lints that only run on the
  rustc source tree itself and are defined in the compiler source like a
  regular built-in lint.

More information about lint registration can be found in the [LintStore]
chapter.

[LintStore]: diagnostics/lintstore.md

### Declaring a lint

The built-in compiler lints are defined in the [`rustc_lint`][builtin]
crate. Lints that need to be implemented in other crates are defined in
[`rustc_lint_defs`]. You should prefer to place lints in `rustc_lint` if
possible. One benefit is that it is close to the dependency root, so it can be
much faster to work on.

[builtin]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/index.html
[`rustc_lint_defs`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint_defs/index.html

Every lint is implemented via a `struct` that implements the `LintPass` `trait`
(you can also implement one of the more specific lint pass traits, either
`EarlyLintPass` or `LateLintPass` depending on when is best for your lint to run).
The trait implementation allows you to check certain syntactic constructs
as the linter walks the AST. You can then choose to emit lints in a
very similar way to compile errors.

You also declare the metadata of a particular lint via the [`declare_lint!`]
macro. This macro includes the name, the default level, a short description, and some
more details.

Note that the lint and the lint pass must be registered with the compiler.

For example, the following lint checks for uses
of `while true { ... }` and suggests using `loop { ... }` instead.

```rust,ignore
// Declare a lint called `WHILE_TRUE`
declare_lint! {
    WHILE_TRUE,

    // warn-by-default
    Warn,

    // This string is the lint description
    "suggest using `loop { }` instead of `while true { }`"
}

// This declares a struct and a lint pass, providing a list of associated lints. The
// compiler currently doesn't use the associated lints directly (e.g., to not
// run the pass or otherwise check that the pass emits the appropriate set of
// lints). However, it's good to be accurate here as it's possible that we're
// going to register the lints via the get_lints method on our lint pass (that
// this macro generates).
declare_lint_pass!(WhileTrue => [WHILE_TRUE]);

// Helper function for `WhileTrue` lint.
// Traverse through any amount of parenthesis and return the first non-parens expression.
fn pierce_parens(mut expr: &ast::Expr) -> &ast::Expr {
    while let ast::ExprKind::Paren(sub) = &expr.kind {
        expr = sub;
    }
    expr
}

// `EarlyLintPass` has lots of methods. We only override the definition of
// `check_expr` for this lint because that's all we need, but you could
// override other methods for your own lint. See the rustc docs for a full
// list of methods.
impl EarlyLintPass for WhileTrue {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &ast::Expr) {
        if let ast::ExprKind::While(cond, ..) = &e.kind
            && let ast::ExprKind::Lit(ref lit) = pierce_parens(cond).kind
            && let ast::LitKind::Bool(true) = lit.kind
            && !lit.span.from_expansion()
        {
            let condition_span = cx.sess.source_map().guess_head_span(e.span);
            cx.struct_span_lint(WHILE_TRUE, condition_span, |lint| {
                lint.build(fluent::example::use_loop)
                    .span_suggestion_short(
                        condition_span,
                        fluent::example::suggestion,
                        "loop".to_owned(),
                        Applicability::MachineApplicable,
                    )
                    .emit();
            })
        }
    }
}
```

```fluent
example-use-loop = denote infinite loops with `loop {"{"} ... {"}"}`
  .suggestion = use `loop`
```

[`declare_lint!`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint_defs/macro.declare_lint.html

### Edition-gated lints

Sometimes we want to change the behavior of a lint in a new edition. To do this,
we just add the transition to our invocation of `declare_lint!`:

```rust,ignore
declare_lint! {
    pub ANONYMOUS_PARAMETERS,
    Allow,
    "detects anonymous parameters",
    Edition::Edition2018 => Warn,
}
```

This makes the `ANONYMOUS_PARAMETERS` lint allow-by-default in the 2015 edition
but warn-by-default in the 2018 edition.

See [Edition-specific lints](./guides/editions.md#edition-specific-lints) for more information.

### Feature-gated lints

Lints belonging to a feature should only be usable if the feature is enabled in the
crate. To support this, lint declarations can contain a feature gate like so:

```rust,ignore
declare_lint! {
    pub SOME_LINT_NAME,
    Warn,
    "a new and useful, but feature gated lint",
    @feature_gate = sym::feature_name;
}
```

### Future-incompatible lints

The use of the term `future-incompatible` within the compiler has a slightly
broader meaning than what rustc exposes to users of the compiler.

Inside rustc, future-incompatible lints are for signalling to the user that code they have
written may not compile in the future. In general, future-incompatible code
exists for two reasons:
* The user has written unsound code that the compiler mistakenly accepted. While
it is within Rust's backwards compatibility guarantees to fix the soundness hole
(breaking the user's code), the lint is there to warn the user that this will happen
in some upcoming version of rustc *regardless of which edition the code uses*. This is the
meaning that rustc exclusively exposes to users as "future incompatible".
* The user has written code that will either no longer compiler *or* will change
meaning in an upcoming *edition*. These are often called "edition lints" and can be
typically seen in the various "edition compatibility" lint groups (e.g., `rust_2021_compatibility`)
that are used to lint against code that will break if the user updates the crate's edition.
See [migration lints](guides/editions.md#migration-lints) for more details.

A future-incompatible lint should be declared with the `@future_incompatible`
additional "field":

```rust,ignore
declare_lint! {
    pub ANONYMOUS_PARAMETERS,
    Allow,
    "detects anonymous parameters",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #41686 <https://github.com/rust-lang/rust/issues/41686>",
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2018),
    };
}
```

Notice the `reason` field which describes why the future incompatible change is happening.
This will change the diagnostic message the user receives as well as determine which
lint groups the lint is added to. In the example above, the lint is an "edition lint"
(since its "reason" is `EditionError`), signifying to the user that the use of anonymous
parameters will no longer compile in Rust 2018 and beyond.

Inside [LintStore::register_lints][fi-lint-groupings], lints with `future_incompatible`
fields get placed into either edition-based lint groups (if their `reason` is tied to
an edition) or into the `future_incompatibility` lint group.

[fi-lint-groupings]: https://github.com/rust-lang/rust/blob/51fd129ac12d5bfeca7d216c47b0e337bf13e0c2/compiler/rustc_lint/src/context.rs#L212-L237

If you need a combination of options that's not supported by the
`declare_lint!` macro, you can always change the `declare_lint!` macro
to support this.

### Renaming or removing a lint

If it is determined that a lint is either improperly named or no longer needed,
the lint must be registered for renaming or removal, which will trigger a warning if a user tries
to use the old lint name. To declare a rename/remove, add a line with
[`store.register_renamed`] or [`store.register_removed`] to the code of the
[`rustc_lint::register_builtins`] function.

```rust,ignore
store.register_renamed("single_use_lifetime", "single_use_lifetimes");
```

[`store.register_renamed`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/struct.LintStore.html#method.register_renamed
[`store.register_removed`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/struct.LintStore.html#method.register_removed
[`rustc_lint::register_builtins`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/fn.register_builtins.html

### Lint groups

Lints can be turned on in groups. These groups are declared in the
[`register_builtins`][rbuiltins] function in [`rustc_lint::lib`][builtin]. The
`add_lint_group!` macro is used to declare a new group.

[rbuiltins]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/fn.register_builtins.html

For example,

```rust,ignore
add_lint_group!(sess,
    "nonstandard_style",
    NON_CAMEL_CASE_TYPES,
    NON_SNAKE_CASE,
    NON_UPPER_CASE_GLOBALS);
```

This defines the `nonstandard_style` group which turns on the listed lints. A
user can turn on these lints with a `!#[warn(nonstandard_style)]` attribute in
the source code, or by passing `-W nonstandard-style` on the command line.

Some lint groups are created automatically in `LintStore::register_lints`. For instance,
any lint declared with `FutureIncompatibleInfo` where the reason is
`FutureIncompatibilityReason::FutureReleaseError` (the default when
`@future_incompatible` is used in `declare_lint!`), will be added to
the `future_incompatible` lint group. Editions also have their own lint groups
(e.g., `rust_2021_compatibility`) automatically generated for any lints signaling
future-incompatible code that will break in the specified edition.

### Linting early in the compiler

On occasion, you may need to define a lint that runs before the linting system
has been initialized (e.g. during parsing or macro expansion). This is
problematic because we need to have computed lint levels to know whether we
should emit a warning or an error or nothing at all.

To solve this problem, we buffer the lints until the linting system is
processed. [`Session`][sessbl] and [`ParseSess`][parsebl] both have
`buffer_lint` methods that allow you to buffer a lint for later. The linting
system automatically takes care of handling buffered lints later.

[sessbl]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/struct.Session.html#method.buffer_lint
[parsebl]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_session/parse/struct.ParseSess.html#method.buffer_lint

Thus, to define a lint that runs early in the compilation, one defines a lint
like normal but invokes the lint with `buffer_lint`.

#### Linting even earlier in the compiler

The parser (`rustc_ast`) is interesting in that it cannot have dependencies on
any of the other `rustc*` crates. In particular, it cannot depend on
`rustc_middle::lint` or `rustc_lint`, where all of the compiler linting
infrastructure is defined. That's troublesome!

To solve this, `rustc_ast` defines its own buffered lint type, which
`ParseSess::buffer_lint` uses. After macro expansion, these buffered lints are
then dumped into the `Session::buffered_lints` used by the rest of the compiler.

## JSON diagnostic output

The compiler accepts an `--error-format json` flag to output
diagnostics as JSON objects (for the benefit of tools such as `cargo
fix`). It looks like this:

```console
$ rustc json_error_demo.rs --error-format json
{"message":"cannot add `&str` to `{integer}`","code":{"code":"E0277","explanation":"\nYou tried to use a type which doesn't implement some trait in a place which\nexpected that trait. Erroneous code example:\n\n```compile_fail,E0277\n// here we declare the Foo trait with a bar method\ntrait Foo {\n    fn bar(&self);\n}\n\n// we now declare a function which takes an object implementing the Foo trait\nfn some_func<T: Foo>(foo: T) {\n    foo.bar();\n}\n\nfn main() {\n    // we now call the method with the i32 type, which doesn't implement\n    // the Foo trait\n    some_func(5i32); // error: the trait bound `i32 : Foo` is not satisfied\n}\n```\n\nIn order to fix this error, verify that the type you're using does implement\nthe trait. Example:\n\n```\ntrait Foo {\n    fn bar(&self);\n}\n\nfn some_func<T: Foo>(foo: T) {\n    foo.bar(); // we can now use this method since i32 implements the\n               // Foo trait\n}\n\n// we implement the trait on the i32 type\nimpl Foo for i32 {\n    fn bar(&self) {}\n}\n\nfn main() {\n    some_func(5i32); // ok!\n}\n```\n\nOr in a generic context, an erroneous code example would look like:\n\n```compile_fail,E0277\nfn some_func<T>(foo: T) {\n    println!(\"{:?}\", foo); // error: the trait `core::fmt::Debug` is not\n                           //        implemented for the type `T`\n}\n\nfn main() {\n    // We now call the method with the i32 type,\n    // which *does* implement the Debug trait.\n    some_func(5i32);\n}\n```\n\nNote that the error here is in the definition of the generic function: Although\nwe only call it with a parameter that does implement `Debug`, the compiler\nstill rejects the function: It must work with all possible input types. In\norder to make this example compile, we need to restrict the generic type we're\naccepting:\n\n```\nuse std::fmt;\n\n// Restrict the input type to types that implement Debug.\nfn some_func<T: fmt::Debug>(foo: T) {\n    println!(\"{:?}\", foo);\n}\n\nfn main() {\n    // Calling the method is still fine, as i32 implements Debug.\n    some_func(5i32);\n\n    // This would fail to compile now:\n    // struct WithoutDebug;\n    // some_func(WithoutDebug);\n}\n```\n\nRust only looks at the signature of the called function, as such it must\nalready specify all requirements that will be used for every type parameter.\n"},"level":"error","spans":[{"file_name":"json_error_demo.rs","byte_start":50,"byte_end":51,"line_start":4,"line_end":4,"column_start":7,"column_end":8,"is_primary":true,"text":[{"text":"    a + b","highlight_start":7,"highlight_end":8}],"label":"no implementation for `{integer} + &str`","suggested_replacement":null,"suggestion_applicability":null,"expansion":null}],"children":[{"message":"the trait `std::ops::Add<&str>` is not implemented for `{integer}`","code":null,"level":"help","spans":[],"children":[],"rendered":null}],"rendered":"error[E0277]: cannot add `&str` to `{integer}`\n --> json_error_demo.rs:4:7\n  |\n4 |     a + b\n  |       ^ no implementation for `{integer} + &str`\n  |\n  = help: the trait `std::ops::Add<&str>` is not implemented for `{integer}`\n\n"}
{"message":"aborting due to previous error","code":null,"level":"error","spans":[],"children":[],"rendered":"error: aborting due to previous error\n\n"}
{"message":"For more information about this error, try `rustc --explain E0277`.","code":null,"level":"","spans":[],"children":[],"rendered":"For more information about this error, try `rustc --explain E0277`.\n"}
```

Note that the output is a series of lines, each of which is a JSON
object, but the series of lines taken together is, unfortunately, not
valid JSON, thwarting tools and tricks (such as [piping to `python3 -m
json.tool`](https://docs.python.org/3/library/json.html#module-json.tool))
that require such. (One speculates that this was intentional for LSP
performance purposes, so that each line/object can be sent as
it is flushed?)

Also note the "rendered" field, which contains the "human" output as a
string; this was introduced so that UI tests could both make use of
the structured JSON and see the "human" output (well, _sans_ colors)
without having to compile everything twice.

The "human" readable and the json format emitter can be found under
`rustc_errors`, both were moved from the `rustc_ast` crate to the
[rustc_errors crate](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/index.html).

The JSON emitter defines [its own `Diagnostic`
struct](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/json/struct.Diagnostic.html)
(and sub-structs) for the JSON serialization. Don't confuse this with
[`errors::Diag`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/struct.Diag.html)!

## `#[rustc_on_unimplemented]`

This attribute allows trait definitions to modify error messages when an implementation was
expected but not found. The string literals in the attribute are format strings and can be
formatted with named parameters. See the Formatting
section below for what parameters are permitted.

```rust,ignore
#[rustc_on_unimplemented(message = "an iterator over \
    elements of type `{A}` cannot be built from a \
    collection of type `{Self}`")]
trait MyIterator<A> {
    fn next(&mut self) -> A;
}

fn iterate_chars<I: MyIterator<char>>(i: I) {
    // ...
}

fn main() {
    iterate_chars(&[1, 2, 3][..]);
}
```

When the user compiles this, they will see the following;

```txt
error[E0277]: an iterator over elements of type `char` cannot be built from a collection of type `&[{integer}]`
  --> src/main.rs:13:19
   |
13 |     iterate_chars(&[1, 2, 3][..]);
   |     ------------- ^^^^^^^^^^^^^^ the trait `MyIterator<char>` is not implemented for `&[{integer}]`
   |     |
   |     required by a bound introduced by this call
   |
note: required by a bound in `iterate_chars`
```

You can modify the contents of:
 - the main error message (`message`)
 - the label (`label`)
 - the note(s) (`note`)

For example, the following attribute

```rust,ignore
#[rustc_on_unimplemented(message = "message", label = "label", note = "note")]
trait MyIterator<A> {
    fn next(&mut self) -> A;
}
```

Would generate the following output:

```text
error[E0277]: message
  --> <file>:10:19
   |
10 |     iterate_chars(&[1, 2, 3][..]);
   |     ------------- ^^^^^^^^^^^^^^ label
   |     |
   |     required by a bound introduced by this call
   |
   = help: the trait `MyIterator<char>` is not implemented for `&[{integer}]`
   = note: note
note: required by a bound in `iterate_chars`
```

The functionality discussed so far is also available with
[`#[diagnostic::on_unimplemented]`](https://doc.rust-lang.org/nightly/reference/attributes/diagnostics.html#the-diagnosticon_unimplemented-attribute).
If you can, you should use that instead.

### Filtering

To allow more targeted error messages, it is possible to filter the
application of these fields with `on`.

You can filter on the following boolean flags:
 - `crate_local`: whether the code causing the trait bound to not be
   fulfilled is part of the user's crate. This is used to avoid suggesting
   code changes that would require modifying a dependency.
 - `direct`: whether this is an user-specified rather than derived obligation.
 - `from_desugaring`: whether we are in some kind of desugaring, like `?`
   or a `try` block for example. This flag can also be matched on, see below.

You can match on the following names and values, using `name = "value"`:
 - `cause`: Match against one variant of the `ObligationCauseCode`
   enum. Only `"MainFunctionType"` is supported.
 - `from_desugaring`: Match against a particular variant of the `DesugaringKind`
   enum. The desugaring is identified by its variant name, for example
   `"QuestionMark"` for `?` desugaring or `"TryBlock"` for `try` blocks.
 - `Self` and any generic arguments of the trait, like `Self = "alloc::string::String"`
   or `Rhs="i32"`.
   
The compiler can provide several values to match on, for example:
  - the self_ty, pretty printed with and without type arguments resolved.
  - `"{integral}"`, if self_ty is an integral of which the type is known.
  - `"[]"`, `"[{ty}]"`, `"[{ty}; _]"`, `"[{ty}; $N]"` when applicable.
  - references to said slices and arrays.
  - `"fn"`, `"unsafe fn"` or `"#[target_feature] fn"` when self is a function.
  - `"{integer}"` and `"{float}"` if the type is a number but we haven't inferred it yet.
  - combinations of the above, like `"[{integral}; _]"`.

For example, the `Iterator` trait can be filtered in the following way:

```rust,ignore
#[rustc_on_unimplemented(
    on(Self = "&str", note = "call `.chars()` or `.as_bytes()` on `{Self}`"),
    message = "`{Self}` is not an iterator",
    label = "`{Self}` is not an iterator",
    note = "maybe try calling `.iter()` or a similar method"
)]
pub trait Iterator {}
```

Which would produce the following outputs:

```text
error[E0277]: `Foo` is not an iterator
 --> src/main.rs:4:16
  |
4 |     for foo in Foo {}
  |                ^^^ `Foo` is not an iterator
  |
  = note: maybe try calling `.iter()` or a similar method
  = help: the trait `std::iter::Iterator` is not implemented for `Foo`
  = note: required by `std::iter::IntoIterator::into_iter`

error[E0277]: `&str` is not an iterator
 --> src/main.rs:5:16
  |
5 |     for foo in "" {}
  |                ^^ `&str` is not an iterator
  |
  = note: call `.chars()` or `.bytes() on `&str`
  = help: the trait `std::iter::Iterator` is not implemented for `&str`
  = note: required by `std::iter::IntoIterator::into_iter`
```

The `on` filter accepts `all`, `any` and `not` predicates similar to the `cfg` attribute:

```rust,ignore
#[rustc_on_unimplemented(on(
    all(Self = "&str", T = "alloc::string::String"),
    note = "you can coerce a `{T}` into a `{Self}` by writing `&*variable`"
))]
pub trait From<T>: Sized {
    /* ... */
}
```

### Formatting 

The string literals are format strings that accept parameters wrapped in braces
but positional and listed parameters and format specifiers are not accepted.
The following parameter names are valid:
- `Self` and all generic parameters of the trait.
- `This`: the name of the trait the attribute is on, without generics.
- `Trait`: the name of the "sugared" trait. See `TraitRefPrintSugared`.
- `ItemContext`: the kind of `hir::Node` we're in, things like `"an async block"`,
   `"a function"`, `"an async function"`, etc.

Something like:

```rust,ignore
#![feature(rustc_attrs)]

#[rustc_on_unimplemented(message = "Self = `{Self}`, \
    T = `{T}`, this = `{This}`, trait = `{Trait}`, \
    context = `{ItemContext}`")]
pub trait From<T>: Sized {
    fn from(x: T) -> Self;
}

fn main() {
    let x: i8 = From::from(42_i32);
}
```

Will format the message into 
```text
"Self = `i8`, T = `i32`, this = `From`, trait = `From<i32>`, context = `a function`"
```
