# Method Checking

In some scenarios we might want to check for methods when developing
a lint. There are two kinds of questions that we might be curious about:

-   Invocation: Does an expression call a specific method?
-   Definition: Does an `impl` define a method?

## Checking if an `expr` is calling a specific method

Suppose we have an `expr`, we can check whether it calls a specific
method, e.g. `our_fancy_method`, by performing a pattern match on
the [`ExprKind`] that we can access from `expr.kind`:

```rust
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_span::sym;
use clippy_utils::is_trait_method;

impl<'tcx> LateLintPass<'tcx> for OurFancyMethodLint {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        // Check our expr is calling a method with pattern matching
        if let hir::ExprKind::MethodCall(path, _, [self_arg, ..], _) = &expr.kind
            // Check if the name of this method is `our_fancy_method`
            && path.ident.name.as_str() == "our_fancy_method"
            // We can check the type of the self argument whenever necessary.
            // (It's necessary if we want to check that method is specifically belonging to a specific trait,
            // for example, a `map` method could belong to user-defined trait instead of to `Iterator`)
            // See the next section for more information.
            && is_trait_method(cx, self_arg, sym::OurFancyTrait)
        {
            println!("`expr` is a method call for `our_fancy_method`");
        }
    }
}
```

Take a closer look at the `ExprKind` enum variant [`MethodCall`] for more
information on the pattern matching. As mentioned in [Define
Lints](defining_lints.md#lint-types), the `methods` lint type is full of pattern
matching with `MethodCall` in case the reader wishes to explore more.

## Checking if a `impl` block implements a method

While sometimes we want to check whether a method is being called or not, other
times we want to know if our `Ty` defines a method.

To check if our `impl` block defines a method `our_fancy_method`, we will
utilize the [`check_impl_item`] method that is available in our beloved
[`LateLintPass`] (for more information, refer to the ["Lint
Passes"](lint_passes.md) chapter in the Clippy book). This method provides us
with an [`ImplItem`] struct, which represents anything within an `impl` block.

Let us take a look at how we might check for the implementation of
`our_fancy_method` on a type:

```rust
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::return_ty;
use rustc_hir::{ImplItem, ImplItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_span::symbol::sym;

impl<'tcx> LateLintPass<'tcx> for MyTypeImpl {
    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx ImplItem<'_>) {
        // Check if item is a method/function
        if let ImplItemKind::Fn(ref signature, _) = impl_item.kind
            // Check the method is named `our_fancy_method`
            && impl_item.ident.name.as_str() == "our_fancy_method"
            // We can also check it has a parameter `self`
            && signature.decl.implicit_self.has_implicit_self()
            // We can go even further and even check if its return type is `String`
            && is_type_diagnostic_item(cx, return_ty(cx, impl_item.hir_id), sym::String)
        {
            println!("`our_fancy_method` is implemented!");
        }
    }
}
```

[`check_impl_item`]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_lint/trait.LateLintPass.html#method.check_impl_item
[`ExprKind`]: https://doc.rust-lang.org/beta/nightly-rustc/rustc_hir/hir/enum.ExprKind.html
[`ImplItem`]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_hir/hir/struct.ImplItem.html
[`LateLintPass`]: https://doc.rust-lang.org/stable/nightly-rustc/rustc_lint/trait.LateLintPass.html
[`MethodCall`]: https://doc.rust-lang.org/beta/nightly-rustc/rustc_hir/hir/enum.ExprKind.html#variant.MethodCall
