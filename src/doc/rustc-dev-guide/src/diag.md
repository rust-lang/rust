# Emitting Diagnostics

A lot of effort has been put into making `rustc` have great error messages.
This chapter is about how to emit compile errors and lints from the compiler.

## `Span`

`Span` is the primary data structure in `rustc` used to represent a location in
the code being compiled. `Span`s are attached to most constructs in HIR and MIR,
allowing for easier error reporting whenever an error comes up.

A `Span` can be looked up in a `CodeMap` to get a "snippet" useful for
displaying errors with [`span_to_snippet` and other similar methods][sptosnip]
on the `CodeMap`.

[sptosnip]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/codemap/struct.CodeMap.html#method.span_to_snippet

## Error messages

The [`rustc_errors`][errors] crate defines most of the utilities used for
reporting errors.

[errors]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/index.html

Most "session"-like types in the compiler (e.g. [`Session`][session]) have
methods (or fields with methods) that allow reporting errors. These methods
usually have names like `span_err` or `struct_span_err` or `span_warn`, etc...
There are lots of them; they emit different types of "errors", such as
warnings, errors, fatal errors, suggestions, etc.

[session]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/session/struct.Session.html

In general, there are two class of such methods: ones that emit an error
directly and ones that allow finer control over what to emit. For example,
[`span_err`][spanerr] emits the given error message at the given `Span`, but
[`struct_span_err`][strspanerr] instead returns a
[`DiagnosticBuilder`][diagbuild].

`DiagnosticBuilder` allows you to add related notes and suggestions to an error
before emitting it by calling the [`emit`][emit] method. See the
[docs][diagbuild] for more info on what you can do.

[spanerr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/session/struct.Session.html#method.span_err
[strspanerr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/session/struct.Session.html#method.struct_span_err
[diagbuild]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/diagnostic_builder/struct.DiagnosticBuilder.html
[emit]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/diagnostic_builder/struct.DiagnosticBuilder.html#method.emit

For example, to add a help message to an error, one might do:

```rust,ignore
let snip = sess.codemap().span_to_snippet(sp);

sess.struct_span_err(sp, "oh no! this is an error!")
    .span_suggestion(other_sp, "try using a qux here", format!("qux {}", snip))
    .emit();
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

## Lints

The compiler linting infrastructure is defined in the [`rustc::lint`][rlint]
module. 

[rlint]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/lint/index.html

### Declaring a lint

The built-in compiler lints are defined in the [`rustc_lint`][builtin]
crate.

[builtin]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/index.html

Each lint is defined as a `struct` that implements the `LintPass` `trait`. The
trait implementation allows you to check certain syntactic constructs the
linter walks the source code. You can then choose to emit lints in a very
similar way to compile errors. Finally, you register the lint to actually get
it to be run by the compiler by using the `declare_lint!` macro.

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

// Define a struct and `impl LintPass` for it.
#[derive(Copy, Clone)]
pub struct WhileTrue;

impl LintPass for WhileTrue {
    fn get_lints(&self) -> LintArray {
        lint_array!(WHILE_TRUE)
    }
}

// LateLintPass has lots of methods. We only override the definition of
// `check_expr` for this lint because that's all we need, but you could
// override other methods for your own lint. See the rustc docs for a full
// list of methods.
impl<'a, 'tcx> LateLintPass<'a, 'tcx> for WhileTrue {
    fn check_expr(&mut self, cx: &LateContext, e: &hir::Expr) {
        if let hir::ExprWhile(ref cond, ..) = e.node {
            if let hir::ExprLit(ref lit) = cond.node {
                if let ast::LitKind::Bool(true) = lit.node {
                    if lit.span.ctxt() == SyntaxContext::empty() {
                        let msg = "denote infinite loops with `loop { ... }`";
                        let condition_span = cx.tcx.sess.codemap().def_span(e.span);
                        let mut err = cx.struct_span_lint(WHILE_TRUE, condition_span, msg);
                        err.span_suggestion_short(condition_span, "use `loop`", "loop".to_owned());
                        err.emit();
                    }
                }
            }
        }
    }
}
```

### Edition Lints

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

### Lint Groups

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
user can turn on these lints by using `!#[warn(nonstandard_style)]`.
