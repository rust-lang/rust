## Adding a new lint

You are probably here because you want to add a new lint to Clippy. If this is
the first time you're contributing to Clippy, this document guides you through
creating an example lint from scratch.

To get started, we will create a lint that detects functions called `foo`,
because that's clearly a non-descriptive name.

* [Setup](#Setup)
* [Testing](#Testing)
* [Rustfix tests](#Rustfix-tests)
* [Edition 2018 tests](#Edition-2018-tests)
* [Lint declaration](#Lint-declaration)
* [Lint passes](#Lint-passes)
* [Emitting a lint](#Emitting-a-lint)
* [Adding the lint logic](#Adding-the-lint-logic)
* [Author lint](#Author-lint)
* [Documentation](#Documentation)
* [Running rustfmt](#Running-rustfmt)
* [Debugging](#Debugging)
* [PR Checklist](#PR-Checklist)
* [Cheatsheet](#Cheatsheet)

### Setup

When working on Clippy, you will need the current git master version of rustc,
which can change rapidly. Make sure you're working near rust-clippy's master,
and use the `setup-toolchain.sh` script to configure the appropriate toolchain
for the Clippy directory.

### Testing

Let's write some tests first that we can execute while we iterate on our lint.

Clippy uses UI tests for testing. UI tests check that the output of Clippy is
exactly as expected. Each test is just a plain Rust file that contains the code
we want to check. The output of Clippy is compared against a `.stderr` file.
Note that you don't have to create this file yourself, we'll get to
generating the `.stderr` files further down.

We start by creating the test file at `tests/ui/foo_functions.rs`. It doesn't
really matter what the file is called, but it's a good convention to name it
after the lint it is testing, so `foo_functions.rs` it is.

Inside the file we put some examples to get started:

```rust
#![warn(clippy::foo_functions)]

// Impl methods
struct A;
impl A {
    pub fn fo(&self) {}
    pub fn foo(&self) {}
    pub fn food(&self) {}
}

// Default trait methods
trait B {
    fn fo(&self) {}
    fn foo(&self) {}
    fn food(&self) {}
}

// Plain functions
fn fo() {}
fn foo() {}
fn food() {}

fn main() {
    // We also don't want to lint method calls
    foo();
    let a = A;
    a.foo();
}

```

Now we can run the test with `TESTNAME=ui/foo_functions cargo uitest`.
Currently this test will fail. If you go through the output you will see that we
are told that `clippy::foo_functions` is an unknown lint, which is expected.

While we are working on implementing our lint, we can keep running the UI
test. That allows us to check if the output is turning into what we want.

Once we are satisfied with the output, we need to run
`tests/ui/update-all-references.sh` to update the `.stderr` file for our lint.
Running `TESTNAME=ui/foo_functions cargo uitest` should pass then. When we
commit our lint, we need to commit the generated `.stderr` files, too.

### Rustfix tests

If the lint you are working on is making use of structured suggestions, the
test file should include a `// run-rustfix` comment at the top. This will
additionally run [rustfix](https://github.com/rust-lang-nursery/rustfix) for
that test. Rustfix will apply the suggestions from the lint to the code of the
test file and compare that to the contents of a `.fixed` file.

Use `tests/ui/update-all-references.sh` to automatically generate the
`.fixed` file after running the tests.

With tests in place, let's have a look at implementing our lint now.

### Edition 2018 tests

Some features require the 2018 edition to work (e.g. `async_await`), but
compile-test tests run on the 2015 edition by default. To change this behavior
add `// compile-flags: --edition 2018` at the top of the test file.

### Testing manually

Manually testing against an example file can be useful if you have added some
`println!`s and the test suite output becomes unreadable. To try Clippy with
your local modifications, run `env CLIPPY_TESTS=true cargo run --bin
clippy-driver -- -L ./target/debug input.rs` from the working copy root.

### Lint declaration

We start by creating a new file in the `clippy_lints` crate. That's the crate
where all the lint code is. We are going to call the file
`clippy_lints/src/foo_functions.rs` and import some initial things we need:

```rust
use rustc::lint::{LintArray, LintPass, EarlyLintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
```

The next step is to provide a lint declaration. Lints are declared using the
[`declare_clippy_lint!`][declare_clippy_lint] macro:

```rust
declare_clippy_lint! {
    pub FOO_FUNCTIONS,
    pedantic,
    "function named `foo`, which is not a descriptive name"
}
```

* `FOO_FUNCTIONS` is the name of our lint. Be sure to follow the [lint naming
guidelines][lint_naming] here when naming your lint. In short, the name should
state the thing that is being checked for and read well when used with
`allow`/`warn`/`deny`.
* `pedantic` sets the lint level to `Allow`.
  The exact mapping can be found [here][category_level_mapping]
* The last part should be a text that explains what exactly is wrong with the
  code

With our lint declaration done, we will now make sure that it is assigned to a
lint pass:

```rust
// clippy_lints/src/foo_functions.rs

// .. imports and lint declaration ..

declare_lint_pass!(FooFunctions => [FOO_FUNCTIONS]);

impl EarlyLintPass for FooFunctions {}
```

Don't worry about the `name` method here. As long as it includes the name of the
lint pass it should be fine.

Next we need to run `util/dev update_lints` to register the lint in various
places, mainly in `clippy_lints/src/lib.rs`.

While `update_lints` automates some things, it doesn't automate everything. We
will have to register our lint pass manually in the `register_plugins` function
in `clippy_lints/src/lib.rs`:

```rust
reg.register_early_lint_pass(box foo_functions::FooFunctions);
```

This should fix the `unknown clippy lint: clippy::foo_functions` error that we
saw when we executed our tests the first time. The next decision we have to make
is which lint pass our lint is going to need.

### Lint passes

Writing a lint that only checks for the name of a function means that we only
have to deal with the AST and don't have to deal with the type system at all.
This is good, because it makes writing this particular lint less complicated.

We have to make this decision with every new Clippy lint. It boils down to using
either [`EarlyLintPass`][early_lint_pass] or [`LateLintPass`][late_lint_pass].

In short, the `LateLintPass` has access to type information while the
`EarlyLintPass` doesn't. If you don't need access to type information, use the
`EarlyLintPass`. The `EarlyLintPass` is also faster. However linting speed
hasn't really been a concern with Clippy so far.

Since we don't need type information for checking the function name, we are
going to use the `EarlyLintPass`. It has to be imported as well, changing our
imports to:

```rust
use rustc::lint::{LintArray, LintPass, EarlyLintPass, EarlyContext};
use rustc::{declare_tool_lint, lint_array};
```

### Emitting a lint

With UI tests and the lint declaration in place, we can start working on the
implementation of the lint logic.

Let's start by implementing the `EarlyLintPass` for our `FooFunctions`:

```rust
impl EarlyLintPass for FooFunctions {
    fn check_fn(&mut self, cx: &EarlyContext<'_>, fn_kind: FnKind<'_>, _: &FnDecl, span: Span, _: NodeId) {
        // TODO: Emit lint here
    }
}
```

We implement the [`check_fn`][check_fn] method from the
[`EarlyLintPass`][early_lint_pass] trait. This gives us access to various
information about the function that is currently being checked. More on that in
the next section. Let's worry about the details later and emit our lint for
*every* function definition first.

Depending on how complex we want our lint message to be, we can choose from a
variety of lint emission functions. They can all be found in
[`clippy_lints/src/utils/diagnostics.rs`][diagnostics].

`span_help_and_lint` seems most appropriate in this case. It allows us to
provide an extra help message and we can't really suggest a better name
automatically. This is how it looks:

```rust
impl EarlyLintPass for FooFunctions {
    fn check_fn(&mut self, cx: &EarlyContext<'_>, _: FnKind<'_>, _: &FnDecl, span: Span, _: NodeId) {
        span_help_and_lint(
            cx,
            FOO_FUNCTIONS,
            span,
            "function named `foo`",
            "consider using a more meaningful name"
        );
    }
}
```

Running our UI test should now produce output that contains the lint message.

### Adding the lint logic

Writing the logic for your lint will most likely be different from our example,
so this section is kept rather short.

Using the [`check_fn`][check_fn] method gives us access to [`FnKind`][fn_kind]
that has two relevant variants for us `FnKind::ItemFn` and `FnKind::Method`.
Both provide access to the name of the function/method via an [`Ident`][ident].

With that we can expand our `check_fn` method to:

```rust
impl EarlyLintPass for FooFunctions {
    fn check_fn(&mut self, cx: &EarlyContext<'_>, fn_kind: FnKind<'_>, _: &FnDecl, span: Span, _: NodeId) {
        if is_foo_fn(fn_kind) {
            span_help_and_lint(
                cx,
                FOO_FUNCTIONS,
                span,
                "function named `foo`",
                "consider using a more meaningful name"
            );
        }
    }
}
```

We separate the lint conditional from the lint emissions because it makes the
code a bit easier to read. In some cases this separation would also allow to
write some unit tests (as opposed to only UI tests) for the separate function.

In our example, `is_foo_fn` looks like:

```rust
// use statements, impl EarlyLintPass, check_fn, ..

fn is_foo_fn(fn_kind: FnKind<'_>) -> bool {
    match fn_kind {
        FnKind::ItemFn(ident, ..) | FnKind::Method(ident, ..) => {
            ident.name == "foo"
        },
        FnKind::Closure(..) => false
    }
}
```

Now we should also run the full test suite with `cargo test`. At this point
running `cargo test` should produce the expected output. Remember to run
`tests/ui/update-all-references.sh` to update the `.stderr` file.

`cargo test` (as opposed to `cargo uitest`) will also ensure that our lint
implementation is not violating any Clippy lints itself.

That should be it for the lint implementation. Running `cargo test` should now
pass.

### Author lint

If you have trouble implementing your lint, there is also the internal `author`
lint to generate Clippy code that detects the offending pattern. It does not
work for all of the Rust syntax, but can give a good starting point.

The quickest way to use it, is the [Rust playground][play].rust-lang.org).
Put the code you want to lint into the editor and add the `#[clippy::author]`
attribute above the item. Then run Clippy via `Tools -> Clippy` and you should
see the generated code in the output below.

[Here][author_example] is an example on the playground.

If the command was executed successfully, you can copy the code over to where
you are implementing your lint.

### Documentation

The final thing before submitting our PR is to add some documentation to our
lint declaration.

Please document your lint with a doc comment akin to the following:

```rust
declare_clippy_lint! {
    /// **What it does:** Checks for ... (describe what the lint matches).
    ///
    /// **Why is this bad?** Supply the reason for linting the code.
    ///
    /// **Known problems:** None. (Or describe where it could go wrong.)
    ///
    /// **Example:**
    ///
    /// ```rust,ignore
    /// // Bad
    /// Insert a short example of code that triggers the lint
    ///
    /// // Good
    /// Insert a short example of improved code that doesn't trigger the lint
    /// ```
    pub FOO_FUNCTIONS,
    pedantic,
    "function named `foo`, which is not a descriptive name"
}
```

Once your lint is merged, this documentation will show up in the [lint
list][lint_list].

### Running rustfmt

[Rustfmt](https://github.com/rust-lang/rustfmt) is a tool for formatting Rust
code according to style guidelines. Your code has to be formatted by `rustfmt`
before a PR can be merged. Clippy uses nightly `rustfmt` in the CI.

It can be installed via `rustup`:

```bash
rustup component add rustfmt --toolchain=nightly
```

Use `./util/dev fmt` to format the whole codebase. Make sure that `rustfmt` is
installed for the nightly toolchain.

### Debugging

If you want to debug parts of your lint implementation, you can use the `dbg!`
macro anywhere in your code. Running the tests should then include the debug
output in the `stdout` part.

### PR Checklist

Before submitting your PR make sure you followed all of the basic requirements:

<!-- Sync this with `.github/PULL_REQUEST_TEMPLATE` -->

- [ ] Followed [lint naming conventions][lint_naming]
- [ ] Added passing UI tests (including committed `.stderr` file)
- [ ] `cargo test` passes locally
- [ ] Executed `./util/dev update_lints`
- [ ] Added lint documentation
- [ ] Run `./util/dev fmt`

### Cheatsheet

Here are some pointers to things you are likely going to need for every lint:

* [Clippy utils][utils] - Various helper functions. Maybe the function you need
  is already in here (`implements_trait`, `match_path`, `snippet`, etc)
* [Clippy diagnostics][diagnostics]
* [The `if_chain` macro][if_chain]
* [`in_macro_or_desugar`][in_macro_or_desugar] and [`in_external_macro`][in_external_macro]
* [`Span`][span]
* [`Applicability`][applicability]
* [The rustc guide][rustc_guide] explains a lot of internal compiler concepts
* [The nightly rustc docs][nightly_docs] which has been linked to throughout
  this guide

For `EarlyLintPass` lints:

* [`EarlyLintPass`][early_lint_pass]
* [`syntax::ast`][ast]

For `LateLintPass` lints:

* [`LateLintPass`][late_lint_pass]
* [`Ty::TyKind`][ty]


While most of Clippy's lint utils are documented, most of rustc's internals lack
documentation currently. This is unfortunate, but in most cases you can probably
get away with copying things from existing similar lints. If you are stuck,
don't hesitate to ask on Discord, IRC or in the issue/PR.

[lint_list]: https://rust-lang.github.io/rust-clippy/master/index.html
[lint_naming]: https://rust-lang.github.io/rfcs/0344-conventions-galore.html#lints
[category_level_mapping]: https://github.com/rust-lang/rust-clippy/blob/bd23cb89ec0ea63403a17d3fc5e50c88e38dd54f/clippy_lints/src/lib.rs#L43
[declare_clippy_lint]: https://github.com/rust-lang/rust-clippy/blob/a71acac1da7eaf667ab90a1d65d10e5cc4b80191/clippy_lints/src/lib.rs#L39
[compilation_stages]: https://rust-lang.github.io/rustc-guide/high-level-overview.html#the-main-stages-of-compilation
[check_fn]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/lint/trait.EarlyLintPass.html#method.check_fn
[early_lint_pass]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/lint/trait.EarlyLintPass.html
[late_lint_pass]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/lint/trait.LateLintPass.html
[fn_kind]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/visit/enum.FnKind.html
[diagnostics]: https://github.com/rust-lang/rust-clippy/blob/master/clippy_lints/src/utils/diagnostics.rs
[utils]: https://github.com/rust-lang/rust-clippy/blob/master/clippy_lints/src/utils/mod.rs
[ident]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/source_map/symbol/struct.Ident.html
[span]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax_pos/struct.Span.html
[applicability]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_errors/enum.Applicability.html
[if_chain]: https://docs.rs/if_chain/0.1.2/if_chain/
[ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty/sty/index.html
[ast]: https://doc.rust-lang.org/nightly/nightly-rustc/syntax/ast/index.html
[in_macro_or_desugar]: https://github.com/rust-lang/rust-clippy/blob/d0717d1f9531a03d154aaeb0cad94c243915a146/clippy_lints/src/utils/mod.rs#L94
[in_external_macro]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/lint/fn.in_external_macro.html
[play]: https://play.rust-lang.org
[author_example]: https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=f093b986e80ad62f3b67a1f24f5e66e2
[rustc_guide]: https://rust-lang.github.io/rustc-guide/
[nightly_docs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/
