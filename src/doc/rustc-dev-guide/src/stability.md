# Stability attributes

This section is about the stability attributes and schemes that allow stable APIs to use unstable
APIs internally in the rustc standard library.

For instructions on stabilizing a language feature see 
[Stabilizing Features](./stabilization_guide.md).

# unstable

The `#[unstable(feature = "foo", issue = "1234", reason = "lorem ipsum")]` attribute explicitly
marks an item as unstable. This infects all sub-items, where the attribute doesn't have to be
reapplied. So if you apply this to a module, all items in the module will be unstable.

You can make specific sub-items stable by using the `#[stable]` attribute on them.
The stability scheme works similarly to how `pub` works. You can have public functions of
nonpublic modules and you can have stable functions in unstable modules or vice versa.

Note, however, that due to a [rustc bug], stable items inside unstable modules
*are* available to stable code in that location!  So, for example, stable code
can import `core::intrinsics::transmute` even though `intrinsics` is an unstable
module.  Thus, this kind of nesting should be avoided when possible.

[rustc bug]: https://github.com/rust-lang/rust/issues/15702

# stable

The `#[stable(feature = "foo", "since = "1.420.69")]` attribute explicitly marks an item as
stabilized. To do this, follow the instructions in
[Stabilizing Features](./stabilization_guide.md).

Note that stable functions may use unstable things in their body.

# allow_internal_unstable

Macros, compiler desugarings and `const fn`s expose their bodies to the call site. To
work around not being able to use unstable things in the standard library's macros, there's the
`#[allow_internal_unstable(feature1, feature2)]` attribute that whitelists the given features for
usage in stable macros or `const fn`s.

Note that `const fn`s are even more special in this regard. You can't just whitelist any feature,
the features need an implementation in `qualify_min_const_fn.rs`. For example the `const_fn_union`
feature gate allows accessing fields of unions inside stable `const fn`s. The rules for when it's
ok to use such a feature gate are that behavior matches the runtime behavior of the same code
(see also [this blog post][blog]). This means that you may not create a
`const fn` that e.g. transmutes a memory address to an integer, because the addresses of things
are nondeterministic and often unknown at compile-time.

Always ping @oli-obk, @RalfJung, and @Centril if you are adding more `allow_internal_unstable`
attributes to any `const fn`

[blog]: https://www.ralfj.de/blog/2018/07/19/const.html
