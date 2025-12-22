# Const Generics

## Kinds of const arguments

Most of the kinds of `ty::Const` that exist have direct parallels to kinds of types that exist, for example `ConstKind::Param` is equivalent to `TyKind::Param`.

The main interesting points here are:
- [`ConstKind::Unevaluated`], this is equivalent to `TyKind::Alias` and in the long term should be renamed (as well as introducing an `AliasConstKind` to parallel `ty::AliasKind`).
- [`ConstKind::Value`], this is the final value of a `ty::Const` after monomorphization. This is similar-ish to fully concrete to things like `TyKind::Str` or `TyKind::ADT`.

For a complete list of *all* kinds of const arguments and how they are actually represented in the type system, see the [`ConstKind`] type.

Inference Variables are quite boring and treated equivalently to type inference variables almost everywhere. Const Parameters are also similarly boring and equivalent to uses of type parameters almost everywhere. However, there are some interesting subtleties with how they are handled during parsing, name resolution, and AST lowering: [ambig-unambig-ty-and-consts].

## Anon Consts

Anon Consts (short for anonymous const items) are how arbitrary expression are represented in const generics, for example an array length of `1 + 1` or `foo()` or even just `0`. These are unique to const generics and have no real type equivalent.

### Desugaring

```rust
struct Foo<const N: usize>;
type Alias = [u8; 1 + 1];
```

In this example we have a const argument of `1 + 1` (the array length) which is represented as an *anon const*. The desugaring would look something like:
```rust
struct Foo<const N: usize>;

const ANON: usize = 1 + 1;
type Alias = [u8; ANON];
```

Where the array length in `[u8; ANON]` isn't itself an anon const containing a usage of `ANON`, but a kind of "direct" usage of the `ANON` const item ([`ConstKind::Unevaluated`]). 

Anon consts do not inherit any generic parameters of the item they are inside of:
```rust
struct Foo<const N: usize>;
type Alias<T: Sized> = [T; 1 + 1];

// Desugars To;

struct Foo<const N: usize>;

const ANON: usize = 1 + 1;
type Alias<T: Sized> = [T; ANON];
```

Note how the `ANON` const has no generic parameters or where clauses, even though `Alias` has both a type parameter `T` and a where clauses `T: Sized`. This desugaring is part of how we enforce that anon consts can't make use of generic parameters.

While it's useful to think of anon consts as being desugared to real const items, the compiler does not actually implement things this way.

At AST lowering time we do not yet know the *type* of the anon const, so we can't desugar to a real HIR item with an explicitly written type. To work around this we have [`DefKind::AnonConst`] and [`hir::Node::AnonConst`] which are used to represent these anonymous const items that can't actually be desugared.

The types of these anon consts are obtainable from the [`type_of`] query. However, the `type_of` query does not actually contain logic for computing the type (infact it just ICEs when called), instead HIR Ty lowering is responsible for *feeding* the value of the `type_of` query for any anon consts that get lowered. HIR Ty lowering can determine the type of the anon const by looking at the type of the Const Parameter that the anon const is an argument to.

TODO: write a chapter on query feeding and link it here

In some sense the desugarings from the previous examples are to:
```rust
struct Foo<const N: usize>;
type Alias = [u8; 1 + 1];

// sort-of desugars to psuedo-rust:
struct Foo<const N: usize>;

const ANON = 1 + 1;
type Alias = [u8; ANON];
```

Where when we go through HIR ty lowering for the array type in `Alias`, we will lower the array length too and feed `type_of(ANON) -> usize`. Effectively setting the type of the `ANON` const item during some later part of the compiler rather than when constructing the HIR. 

After all of this desugaring has taken place the final representation in the type system (ie as a `ty::Const`) is a `ConstKind::Unevaluated` with the `DefId` of the `AnonConst`. This is equivalent to how we would representa a usage of an actual const item if we were to represent them without going through an anon const (e.g. when `min_generic_const_args` is enabled).

This allows the representation for const "aliases" to be the same as the representation of `TyKind::Alias`. Having a proper HIR body also allows for a *lot* of code re-use, e.g. we can reuse HIR typechecking and all of the lowering steps to MIR where we can then reuse const eval. 

### Enforcing lack of Generic Parameters

There are three ways that we enforce anon consts can't use generic parameters:
1. Name Resolution will not resolve paths to generic parameters when inside of an anon const
2. HIR Ty lowering will error when a `Self` type alias to a type referencing generic parameters is encountered inside of an anon const
3. Anon Consts do not inherit where clauses or generics from their parent definition (ie [`generics_of`] does not contain a parent for anon consts)

```rust
// *1* Errors in name resolution
type Alias<const N: usize> = [u8; N + 1];
//~^ ERROR: generic parameters may not be used in const operations

// *2* Errors in HIR Ty lowering:
struct Foo<T>(T);
impl<T> Foo<T> {
    fn assoc() -> [u8; { let a: Self; 0 }] {}
    //~^ ERROR: generic `Self` types are currently not permitted in anonymous constants
}

// *3* Errors due to lack of where clauses on the desugared anon const
trait Trait<T> {
    const ASSOC: usize;
}
fn foo<T>() -> [u8; <()>::ASSOC]
//~^ ERROR: no associated item named `ASSOC` found for unit type `()`
where
    (): Trait<T> {}
```

The second point is particularly subtle as it is very easy to get HIR Ty lowering wrong and not properly enforce that anon consts can't use generic parameters. The existing check is too conservative and accidentally permits some generic parameters to wind up in the body of the anon const [#144547](https://github.com/rust-lang/rust/issues/144547).

Erroneously allowing generic parameters in anon consts can sometimes lead to ICEs but can also lead to accepting illformed programs.

The third point is also somewhat subtle, by not inheriting any of the where clauses of the parent item we can't wind up with the trait solving inferring inference variables to generic parameters based off where clauses in scope that mention generic parameters. For example inferring `?x=T` from the expression `<() as Trait<?x>>::ASSOC` and an in scope where clause of `(): Trait<T>`.

This also makes it much more likely that the compiler will ICE or atleast incidentally emit some kind of error if we *do* accidentally allow generic parameters in an anon const, as the anon const will have none of the necessary information in its environment to properly handle the generic parameters.

```rust
fn foo<T: Sized>() {
    let a = [1_u8;  size_of::<*mut T>()];
}
```

The one exception to all of the above is repeat counts of array expressions. As a *backwards compatibility hack* we allow the repeat count const argument to use generic parameters.

However, to avoid most of the problems involved in allowing generic parameters in anon const const arguments we require that the constant be evaluated before monomorphization (e.g. during type checking). In some sense we only allow generic parameters here when they are semantically unused.

In the previous example the anon const can be evaluated for any type parameter `T` because raw pointers to sized types always have the same size (e.g. `8` on 64bit platforms).

When detecting that we evaluated an anon const that syntactically contained generic parameters, but did not actually depend on them for evaluation to succeed, we emit the [`const_evaluatable_unchecked` FCW][cec_fcw]. This is intended to become a hard error once we stabilize more ways of using generic parameters in const arguments, for example `min_generic_const_args` or (the now dead) `generic_const_exprs`.

The implementation for this FCW can be found here: [`const_eval_resolve_for_typeck`]

### Incompatibilities with `generic_const_parameter_types`

Supporting const paramters such as `const N: [u8; M]` or `const N: Foo<T>` does not work very nicely with the current anon consts setup. There are two reasons for this:
1. As anon consts cannot use generic parameters, their type *also* can't reference generic parameters. This means it is fundamentally not possible to use an anon const as an argument to a const parameeter whose type still references generic parameters.

    ```rust
    #![feature(adt_const_params, generic_const_parameter_types)]

    fn foo<const N: usize, const M: [u8; N]>() {}

    fn bar<const N: usize>() {
        // There is no way to specify the const argument to `M`
        foo::<N, { [1_u8; N] }>();
    }
    ```

2. We currently require knowing the type of anon consts when lowering them during HIR ty lowering. With generic const parameter types it may be the case that the currently known type contains inference variables (ie may not be fully known yet). 

    ```rust
    #![feature(adt_const_params, generic_const_parameter_types)]

    fn foo<const N: usize, const M: [u8; N]>() {}

    fn bar() {
        // The const argument to `N` must be explicitly specified
        // even though it is able to be inferred
        foo::<_, { [1_u8; 3] }>();
    }
    ```

It is currently unclear what the right way to make `generic_const_parameter_types` work nicely with the rest of const generics is. 

`generic_const_exprs` would have allowed for anon consts with types referencing generic parameters, but that design wound up unworkable. 

`min_generic_const_args` will allow for some expressions (for example array construction) to be representable without an anon const and therefore without running into these issues, though whether this is *enough* has yet to be determined.

## Checking types of Const Arguments

In order for a const argument to be well formed it must have the same type as the const parameter it is an argument to. For example a const argument of type `bool` for an array length is not well formed, as an array's length parameter has type `usize`.

```rust
type Alias<const B: bool> = [u8; B];
//~^ ERROR: 
```

To check this we have [`ClauseKind::ConstArgHasType(ty::Const, Ty)`][const_arg_has_type], where for each Const Parameter defined on an item we also desugar an equivalent `ConstArgHasType` clause into its list of where cluases. This ensures that whenever we check wellformedness of anything by proving all of its clauses, we also check happen to check that all of the Const Arguments have the correct type.

```rust
fn foo<const N: usize>() {}

// desugars to in psuedo-rust

fn foo<const N>()
where
//  ConstArgHasType(N, usize)
    N: usize, {}
```

Proving `ConstArgHasType` goals is implemented by first computing the type of the const argument, then equating it with the provided type. A rough outline of how the type of a Const Argument may be computed:
- [`ConstKind::Param(N)`][`ConstKind::Param`] can be looked up in the [`ParamEnv`] to find a `ConstArgHasType(N, ty)` clause
- [`ConstKind::Value`] stores the type of the value inside itself so can trivially be accessed
- [`ConstKind::Unevaluated`] can have its type computed by calling the `type_of` query
- See the implementation of proving `ConstArgHasType` goals for more detailed information

`ConstArgHasType` is *the* soundness critical way that we check Const Arguments have the correct type. However, we do *indirectly* check the types of Const Arguments a different way in some cases.

```rust
type Alias = [u8; true];

// desugars to

const ANON: usize = true;
type Alias = [u8; ANON];
```

By feeding the type of an anon const with the type of the Const Parameter we guarantee that the `ConstArgHasType` goal involving the anon const will succeed. In cases where the type of the anon const doesn't match the type of the Const Parameter what actually happens is a *type checking* error when type checking the anon const's body.

Looking at the above example, this corresponds to `[u8; ANON]` being a well formed type because `ANON` has type `usize`, but the *body* of `ANON` being illformed and resulting in a type checking error because `true` can't be returned from a const item of type `usize`.

[ambig-unambig-ty-and-consts]: ./ambig-unambig-ty-and-consts.md
[`ConstKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.ConstKind.html
[`ConstKind::Infer`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.ConstKind.html#variant.Infer
[`ConstKind::Param`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.ConstKind.html#variant.Param
[`ConstKind::Unevaluated`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.ConstKind.html#variant.Unevaluated
[`ConstKind::Value`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.ConstKind.html#variant.Value
[const_arg_has_type]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.ClauseKind.html#variant.ConstArgHasType
[`ParamEnv`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html
[`generics_of`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#impl-TyCtxt%3C'tcx%3E/method.generics_of
[`type_of`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#method.type_of
[`DefKind::AnonConst`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def/enum.DefKind.html#variant.AnonConst
[`hir::Node::AnonConst`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/hir/enum.Node.html#variant.AnonConst#
[cec_fcw]: https://github.com/rust-lang/rust/issues/76200
[`const_eval_resolve_for_typeck`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#method.const_eval_resolve_for_typeck
