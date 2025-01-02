# Generic parameter definitions

This chapter will discuss how rustc tracks what generic parameters are introduced. For example given some `struct Foo<T>` how does rustc track that `Foo` defines some type parameter `T` (and no other generic parameters).

This will *not* cover how we track generic parameters introduced via `for<'a>` syntax (e.g. in where clauses or `fn` types), which is covered elsewhere in the [chapter on `Binder`s ][ch_binders].

# `ty::Generics`

The generic parameters introduced by an item are tracked by the [`ty::Generics`] struct. Sometimes items allow usage of generics defined on parent items, this is accomplished via the `ty::Generics` struct having an optional field to specify a parent item to inherit generic parameters of. For example given the following code:

```rust,ignore
trait Trait<T> {
    fn foo<U>(&self);
}
```

The `ty::Generics` used for `foo` would contain `[U]` and a parent of `Some(Trait)`. `Trait` would have a `ty::Generics` containing `[Self, T]` with a parent of `None`.

The [`GenericParamDef`] struct is used to represent each individual generic parameter in a `ty::Generics` listing. The `GenericParamDef` struct contains information about the generic parameter, for example its name, defid, what kind of parameter it is (i.e. type, const, lifetime). 

`GenericParamDef` also contains a `u32` index representing what position the parameter is (starting from the outermost parent), this is the value used to represent usages of generic parameters (more on this in the [chapter on representing types][ch_representing_types]).

Interestingly, `ty::Generics` does not currently contain _every_ generic parameter defined on an item. In the case of functions it only contains the _early bound_ parameters.

[ch_representing_types]: ./ty.md
[`ty::Generics`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Generics.html
[`GenericParamDef`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/generics/struct.GenericParamDef.html

# Early vs Late bound parameters


```rust
fn foo<'a, T>(b: &'a T) -> &'a T { b }
//     ^^  ^early bound
//     ^^
//     ^^late bound
```

Generally when referring to an item with generic parameters you must specify a list of generic arguments corresponding to the item's generic parameters. In some cases it is permitted to elide these arguments but still, implicitly, a set of arguments are provided (e.g. `Vec::default()` desugars to `Vec::<_>::default()`).

For functions this is not necessarily the case, for example if we take the function `foo` from the example above and write the following code:
```rust
fn main() {
    let f = foo::<_>;

    let b = String::new();
    let c = String::new();
    
    f(&b);
    drop(b);
    f(&c);
}
```

This code compiles perfectly fine even though there is no single lifetime that could possibly be specified in `foo::<_>` that would allow for both
the `&b` and `&c` borrows to be used as arguments (note: the `drop(b)` line forces the `&b` borrow to be shorter than the `&c` borrow). This works because the `'a` lifetime is _late bound_.

A generic parameter being late bound means that when we write `foo::<_>` we do not actually provide an argument for that parameter, instead we wait until _calling_ the function to provide the generic argument. In the above example this means that we are doing something like `f::<'_>(&b);` and `f::<'_>(&c);` (although in practice we do not actually support turbofishing late bound parameters in this manner)

It may be helpful to think of "early bound parameter" or "late bound parameter" as meaning "early provided parameter" and "late provided parameter", i.e. we provide the argument to the parameter either early (when naming the function) or late (when calling it).

Late bound parameters on functions are tracked with a [`Binder`] when accessing the signature of the function, this can be done with the [`fn_sig`] query. For more information of binders see the [chapter on `Binder`s ][ch_binders].

[`Binder`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/binder/struct.Binder.html
[`fn_sig`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.fn_sig
[ch_binders]: ./ty_module/binders.md