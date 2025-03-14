# Type Checking

When we work on a new lint or improve an existing lint, we might want
to retrieve the type `Ty` of an expression `Expr` for a variety of
reasons. This can be achieved by utilizing the [`LateContext`][LateContext]
that is available for [`LateLintPass`][LateLintPass].

## `LateContext` and `TypeckResults`

The lint context [`LateContext`][LateContext] and [`TypeckResults`][TypeckResults]
(returned by `LateContext::typeck_results`) are the two most useful data structures
in `LateLintPass`. They allow us to jump to type definitions and other compilation
stages such as HIR.

> Note: `LateContext.typeck_results`'s return value is [`TypeckResults`][TypeckResults]
> and is created in the type checking step, it includes useful information such as types of
> expressions, ways to resolve methods and so on.

`TypeckResults` contains useful methods such as [`expr_ty`][expr_ty],
which gives us access to the underlying structure [`Ty`][Ty] of a given expression.

```rust
pub fn expr_ty(&self, expr: &Expr<'_>) -> Ty<'tcx>
```

As a side note, besides `expr_ty`, [`TypeckResults`][TypeckResults] contains a
[`pat_ty()`][pat_ty] method that is useful for retrieving a type from a pattern.

## `Ty`

`Ty` struct contains the type information of an expression.
Let's take a look at `rustc_middle`'s [`Ty`][Ty] struct to examine this struct:

```rust
pub struct Ty<'tcx>(Interned<'tcx, WithStableHash<TyS<'tcx>>>);
```

At a first glance, this struct looks quite esoteric. But at a closer look,
we will see that this struct contains many useful methods for type checking.

For instance, [`is_char`][is_char] checks if the given `Ty` struct corresponds
to the primitive character type.

### `is_*` Usage

In some scenarios, all we need to do is check if the `Ty` of an expression
is a specific type, such as `char` type, so we could write the following:

```rust
impl LateLintPass<'_> for MyStructLint {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        // Get type of `expr`
        let ty = cx.typeck_results().expr_ty(expr);

        // Check if the `Ty` of this expression is of character type
        if ty.is_char() {
            println!("Our expression is a char!");
        }
    }
}
```

Furthermore, if we examine the [source code][is_char_source] for `is_char`,
we find something very interesting:

```rust
#[inline]
pub fn is_char(self) -> bool {
    matches!(self.kind(), Char)
}
```

Indeed, we just discovered `Ty`'s [`kind()` method][kind], which provides us
with [`TyKind`][TyKind] of a `Ty`.

## `TyKind`

`TyKind` defines the kinds of types in Rust's type system.
Peeking into [`TyKind` documentation][TyKind], we will see that it is an
enum of over 25 variants, including items such as `Bool`, `Int`, `Ref`, etc.

### `kind` Usage

The `TyKind` of `Ty` can be returned by calling [`Ty.kind()` method][kind].
We often use this method to perform pattern matching in Clippy.

For instance, if we want to check for a `struct`, we could examine if the
`ty.kind` corresponds to an [`Adt`][Adt] (algebraic data type) and if its
[`AdtDef`][AdtDef] is a struct:

```rust
impl LateLintPass<'_> for MyStructLint {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        // Get type of `expr`
        let ty = cx.typeck_results().expr_ty(expr);
        // Match its kind to enter the type
        match ty.kind() {
            ty::Adt(adt_def, _) if adt_def.is_struct() => println!("Our `expr` is a struct!"),
            _ => ()
        }
    }
}
```

## `hir::Ty` and `ty::Ty`

We've been talking about [`ty::Ty`][middle_ty] this whole time without addressing [`hir::Ty`][hir_ty], but the latter
is also important to understand.

`hir::Ty` would represent *what* the user wrote, while `ty::Ty` is how the compiler sees the type and has more
information. Example:

```rust
fn foo(x: u32) -> u32 { x }
```

Here the HIR sees the types without "thinking" about them, it knows that the function takes an `u32` and returns
an `u32`. As far as `hir::Ty` is concerned those might be different types. But at the `ty::Ty` level the compiler
understands that they're the same type, in-depth lifetimes, etc...

To get from a `hir::Ty` to a `ty::Ty`, you can use the [`lower_ty`][lower_ty] function outside of bodies or
the [`TypeckResults::node_type()`][node_type] method inside of bodies.

> **Warning**: Don't use `lower_ty` inside of bodies, because this can cause ICEs.

## Creating Types programmatically

A common usecase for creating types programmatically is when we want to check if a type implements a trait (see
[Trait Checking](trait_checking.md)).

Here's an example of how to create a `Ty` for a slice of `u8`, i.e. `[u8]`

```rust
use rustc_middle::ty::Ty;
// assume we have access to a LateContext
let ty = Ty::new_slice(cx.tcx, Ty::new_u8());
```

In general, we rely on `Ty::new_*` methods. These methods define the basic building-blocks that the
type-system and trait-system use to define and understand the written code.

## Useful Links

Below are some useful links to further explore the concepts covered
in this chapter:

- [Stages of compilation](https://rustc-dev-guide.rust-lang.org/compiler-src.html#the-main-stages-of-compilation)
- [Diagnostic items](https://rustc-dev-guide.rust-lang.org/diagnostics/diagnostic-items.html)
- [Type checking](https://rustc-dev-guide.rust-lang.org/type-checking.html)
- [Ty module](https://rustc-dev-guide.rust-lang.org/ty.html)

[Adt]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/ty_kind/enum.TyKind.html#variant.Adt
[AdtDef]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/adt/struct.AdtDef.html
[expr_ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TypeckResults.html#method.expr_ty
[node_type]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TypeckResults.html#method.node_type
[is_char]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Ty.html#method.is_char
[is_char_source]: https://doc.rust-lang.org/nightly/nightly-rustc/src/rustc_middle/ty/sty.rs.html#1831-1834
[kind]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Ty.html#method.kind
[LateContext]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/struct.LateContext.html
[LateLintPass]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_lint/trait.LateLintPass.html
[pat_ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/typeck_results/struct.TypeckResults.html#method.pat_ty
[Ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Ty.html
[TyKind]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/ty_kind/enum.TyKind.html
[TypeckResults]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TypeckResults.html
[middle_ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Ty.html
[hir_ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/struct.Ty.html
[lower_ty]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/fn.lower_ty.html
