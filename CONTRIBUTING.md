# Contributing to rust-clippy

Hello fellow Rustacean! Great to see your interest in compiler internals and lints!

## Getting started

All issues on Clippy are mentored, if you want help with a bug just ask @Manishearth or @llogiq.

Some issues are easier than others. The [E-easy](https://github.com/Manishearth/rust-clippy/labels/E-easy)
label can be used to find the easy issues. If you want to work on an issue, please leave a comment
so that we can assign it to you!

Issues marked [T-AST](https://github.com/Manishearth/rust-clippy/labels/T-AST) involve simple
matching of the syntax tree structure, and are generally easier than
[T-middle](https://github.com/Manishearth/rust-clippy/labels/T-middle) issues, which involve types
and resolved paths.

Issues marked [E-medium](https://github.com/Manishearth/rust-clippy/labels/E-medium) are generally
pretty easy too, though it's recommended you work on an E-easy issue first.

[Llogiq's blog post on lints](https://llogiq.github.io/2015/06/04/workflows.html) is a nice primer
to lint-writing, though it does get into advanced stuff. Most lints consist of an implementation of
`LintPass` with one or more of its default methods overridden. See the existing lints for examples
of this.

T-AST issues will generally need you to match against a predefined syntax structure. To figure out
how this syntax structure is encoded in the AST, it is recommended to run `rustc -Z ast-json` on an
example of the structure and compare with the
[nodes in the AST docs](http://manishearth.github.io/rust-internals-docs/syntax/ast/). Usually
the lint will end up to be a nested series of matches and ifs,
[like so](https://github.com/Manishearth/rust-clippy/blob/de5ccdfab68a5e37689f3c950ed1532ba9d652a0/src/misc.rs#L34).

T-middle issues can be more involved and require verifying types. The
[`middle::ty`](http://manishearth.github.io/rust-internals-docs/rustc/middle/ty) module contains a
lot of methods that are useful, though one of the most useful would be `expr_ty` (gives the type of
an AST expression). `match_def_path()` in Clippy's `utils` module can also be useful.

Should you add a lint, try it on clippy itself using `util/dogfood.sh`. You may find that clippy
contains some questionable code itself! Also before making a pull request, please run
`util/update_lints.py`, which will update `lib.rs` and `README.md` with the lint declarations. Our
travis build actually checks for this.

Also please document your lint with a doc comment akin to the following:
```
/// **What it does:** Describe what the lint matches.
///
/// **Why is this bad?** Write the reason for linting the code.
///
/// **Known problems:** Hopefully none.
///
/// **Example:** Insert a short example if you have one
```

Our `util/update_wiki.py` script can then add your lint docs to the wiki.

## Contributions

Clippy welcomes contributions from everyone.

Contributions to Clippy should be made in the form of GitHub pull requests. Each pull request will
be reviewed by a core contributor (someone with permission to land patches) and either landed in the
main tree or given feedback for changes that would be required.

All code in this repository is under the [Mozilla Public License, 2.0](https://www.mozilla.org/MPL/2.0/)

## Conduct

We follow the [Rust Code of Conduct](http://www.rust-lang.org/conduct.html).


<!-- adapted from https://github.com/servo/servo/blob/master/CONTRIBUTING.md -->
