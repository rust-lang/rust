# What is `ty::Generics`

The generic parameters introduced by an item are tracked the [`ty::Generics`] struct. Sometimes items allows usage of generics of parent items inside of them, this is accomplished via the `ty::Generics` struct having an optional field to specify a parent item to inherit generic parameters of. For example given the following code:

```rust,ignore
trait Trait<T> {
    fn foo<U>(&self);
}
```

The `ty::Generics` used for `foo` would contain `[U]` and a parent of `Some(Trait)`. `Trait` would have a `ty::Generics` containing `[Self, T]` with a parent of `None`.

The [`GenericParamDef`] struct is used to represent each individual generic parameter in a `ty::Generics` listing. The `GenericParamDef` struct contains information about the generic parameter, for example its name, defid, what kind of parameter it is (i.e. type, const, lifetime). 

`GenericParamDef` also contains a `u32` index representing what position the parameter is (starting from the outermost parent), this is the value used to represent usages of generic parameters (more on this in the [chapter on representing types][ch_representing_types]).

Interestingly, `ty::Generics` does not currently contain _every_ generic parameter defined on an item. In the case of functions it only contains the _early bound_ lifetime parameters. See the next chapter for information on what "early bound" and "late bound" parameters are.

[ch_representing_types]: ./ty.md
[`ty::Generics`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Generics.html
[`GenericParamDef`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/generics/struct.GenericParamDef.html