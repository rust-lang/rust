# Trait Checking

Besides [type checking](type_checking.md), we might want to examine if
a specific type `Ty` implements certain trait when implementing a lint.
There are three approaches to achieve this, depending on if the target trait
that we want to examine has a [diagnostic item][diagnostic_items],
[lang item][lang_items], or neither.

## Using Diagnostic Items

As explained in the [Rust Compiler Development Guide][rustc_dev_guide], diagnostic items
are introduced for identifying types via [Symbols][symbol].

While the Rust Compiler Development Guide has [a section][using_diagnostic_items] on
how to check for a specific trait on a type `Ty`, Clippy provides
a helper function `is_trait_method`, which simplifies the process for us.

For instance, if we want to examine whether an expression implements
the `Iterator` trait, we could simply write the following code,
providing the `LateContext` (`cx`), our expression at hand, and
the symbol of the trait in question:

```rust
use clippy_utils::is_trait_method;
use rustc_hir::Expr;
use rustc_lint::{LateContext, LateLintPass};
use rustc_span::symbol::sym;

impl LateLintPass<'_> for CheckIteratorTraitLint {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if is_trait_method(cx, expr, sym::Iterator) {
            println!("This expression implements `Iterator` trait!");
        }
    }
}
```

> **Note**: Refer to [this index][symbol_index] for all the defined `Symbol`s.

## Using Lang Items

Besides diagnostic items, we can also use [`lang_items`][lang_items].
Take a look at the documentation and we find that `LanguageItems` contains
all language items both from the current crate or its
dependencies.

Using one of its `*_trait` method, we could obtain the [DefId] of any
specific item, such as `Clone`, `Copy`, `Drop`, `Eq`, which are familiar
to many Rustaceans.

For instance, if we want to examine whether an expression `expr` implements
`Drop` trait, we could access `LanguageItems` via our `LateContext`'s
[TyCtxt], which provides a `lang_items` method that will return the id of
`Drop` trait to us. Then, by calling Clippy utils function `implements_trait`
we can check that the `Ty` of the `expr` implements the trait:

```rust
use clippy_utils::implements_trait;
use rustc_hir::Expr;
use rustc_lint::{LateContext, LateLintPass};

impl LateLintPass<'_> for CheckDropTraitLint {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let ty = cx.typeck_results().expr_ty(expr);
        if cx.tcx.lang_items()
            .drop_trait()
            .map_or(false, |id| implements_trait(cx, ty, id, &[])) {
                println!("`expr` implements `Drop` trait!");
            }
    }
}
```

## Using Type Path

If neither diagnostic item or lang item is available, we can use
[`clippy_utils::paths`][paths] with the `match_trait_method` to determine trait
implementation.

> **Note**: This approach should be avoided if possible.

Below, we check if the given `expr` implements `tokio`'s
[`AsyncReadExt`][AsyncReadExt] trait:

```rust
use clippy_utils::{match_trait_method, paths};
use rustc_hir::Expr;
use rustc_lint::{LateContext, LateLintPass};

impl LateLintPass<'_> for CheckTokioAsyncReadExtTrait {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if match_trait_method(cx, expr, &paths::TOKIO_IO_ASYNCREADEXT) {
            println!("`expr` implements `TOKIO_IO_ASYNCREADEXT` trait!");
        }
    }
}
```

> **Note**: Even though all the `clippy_utils` methods we have seen in this
> chapter takes `expr` as a parameter, these methods are actually using
> each expression's `HirId` under the hood.

[AsyncReadExt]: https://docs.rs/tokio/latest/tokio/io/trait.AsyncReadExt.html
[DefId]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.DefId.html
[diagnostic_items]: https://rustc-dev-guide.rust-lang.org/diagnostics/diagnostic-items.html
[lang_items]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/lang_items/struct.LanguageItems.html
[paths]: https://github.com/rust-lang/rust-clippy/blob/master/clippy_utils/src/paths.rs
[rustc_dev_guide]: https://rustc-dev-guide.rust-lang.org/
[symbol]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/symbol/struct.Symbol.html
[symbol_index]: https://doc.rust-lang.org/beta/nightly-rustc/rustc_span/symbol/sym/index.html
[TyCtxt]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html
[using_diagnostic_items]: https://rustc-dev-guide.rust-lang.org/diagnostics/diagnostic-items.html#using-diagnostic-items
