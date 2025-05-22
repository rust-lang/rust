# Trait Checking

Besides [type checking](type_checking.md), we might want to examine if
a specific type `Ty` implements certain trait when implementing a lint.
There are three approaches to achieve this, depending on if the target trait
that we want to examine has a [diagnostic item][diagnostic_items],
[lang item][lang_items], or neither.

## Using Diagnostic Items

As explained in the [Rust Compiler Development Guide][rustc_dev_guide], diagnostic items
are introduced for identifying types via [Symbols][symbol].

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
		let implements_iterator = cx.tcx.get_diagnostic_item(sym::Iterator).map_or(false, |id| {
    		implements_trait(cx, cx.typeck_results().expr_ty(arg), id, &[])
		});
		if implements_iterator {
			// [...]
		}

    }
}
```

> **Note**: Refer to [this index][symbol_index] for all the defined `Symbol`s.

## Using Lang Items

Besides diagnostic items, we can also use [`lang_items`][lang_items].
Take a look at the documentation to find that `LanguageItems` contains
all language items defined in the compiler.

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

If neither diagnostic item nor a language item is available, we can use
[`clippy_utils::paths`][paths] to determine get a trait's `DefId`.

> **Note**: This approach should be avoided if possible, the best thing to do would be to make a PR to [`rust-lang/rust`][rust] adding a diagnostic item.

Below, we check if the given `expr` implements [`core::iter::Step`](https://doc.rust-lang.org/std/iter/trait.Step.html):

```rust
use clippy_utils::{implements_trait, paths};
use rustc_hir::Expr;
use rustc_lint::{LateContext, LateLintPass};

impl LateLintPass<'_> for CheckIterStep {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let ty = cx.typeck_results().expr_ty(expr);
        if let Some(trait_def_id) = paths::ITER_STEP.first(cx)
            && implements_trait(cx, ty, trait_def_id, &[])
        {
            println!("`expr` implements the `core::iter::Step` trait!");
        }
    }
}
```

## Creating Types Programmatically

Traits are often generic over a type parameter, e.g. `Borrow<T>` is generic
over `T`. Rust allows us to implement a trait for a specific type. For example,
we can implement `Borrow<[u8]>` for a hypothetical type `Foo`. Let's suppose
that we would like to find whether our type actually implements `Borrow<[u8]>`.

To do so, we can use the same `implements_trait` function as above, and supply
a type parameter that represents `[u8]`. Since `[u8]` is a specialization of
`[T]`, we can use the  [`Ty::new_slice`][new_slice] method to create a type
that represents `[T]` and supply `u8` as a type parameter.
To create a `ty::Ty` programmatically, we rely on `Ty::new_*` methods. These
methods create a `TyKind` and then wrap it in a `Ty` struct. This means we
have access to all the primitive types, such as `Ty::new_char`,
`Ty::new_bool`, `Ty::new_int`, etc. We can also create more complex types,
such as slices, tuples, and references out of these basic building blocks.

For trait checking, it is not enough to create the types, we need to convert
them into [GenericArg]. In rustc, a generic is an entity that the compiler
understands and has three kinds, type, const and lifetime. By calling
`.into()` on a constructed [Ty], we wrap the type into a generic which can
then be used by the query system to decide whether the specialized trait
is implemented.

The following code demonstrates how to do this:

```rust

use rustc_middle::ty::Ty;
use clippy_utils::ty::implements_trait;
use rustc_span::symbol::sym;

let ty = todo!("Get the `Foo` type to check for a trait implementation");
let borrow_id = cx.tcx.get_diagnostic_item(sym::Borrow).unwrap(); // avoid unwrap in real code
let slice_of_bytes_t = Ty::new_slice(cx.tcx, cx.tcx.types.u8);
let generic_param = slice_of_bytes_t.into();
if implements_trait(cx, ty, borrow_id, &[generic_param]) {
    todo!("Rest of lint implementation")
}
```

In essence, the [Ty] struct allows us to create types programmatically in a
representation that can be used by the compiler and the query engine. We then
use the `rustc_middle::Ty` of the type we are interested in, and query the
compiler to see if it indeed implements the trait we are interested in.


[DefId]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.DefId.html
[diagnostic_items]: https://rustc-dev-guide.rust-lang.org/diagnostics/diagnostic-items.html
[lang_items]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/lang_items/struct.LanguageItems.html
[paths]: https://github.com/rust-lang/rust-clippy/blob/master/clippy_utils/src/paths.rs
[rustc_dev_guide]: https://rustc-dev-guide.rust-lang.org/
[symbol]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/symbol/struct.Symbol.html
[symbol_index]: https://doc.rust-lang.org/beta/nightly-rustc/rustc_span/symbol/sym/index.html
[TyCtxt]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html
[Ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Ty.html
[rust]: https://github.com/rust-lang/rust
[new_slice]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Ty.html#method.new_slice
[GenericArg]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.GenericArg.html
