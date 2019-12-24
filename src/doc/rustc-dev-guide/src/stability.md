# Stability attributes

This section is about the stability attributes and schemes that allow stable
APIs to use unstable APIs internally in the rustc standard library.

For instructions on stabilizing a language feature see [Stabilizing
Features](./stabilization_guide.md).

## unstable

The `#[unstable(feature = "foo", issue = "1234", reason = "lorem ipsum")]`
attribute explicitly marks an item as unstable. Items that are marked as
"unstable" cannot be used without a corresponding `#![feature]` attribute on
the crate, even on a nightly compiler. This restriction only applies across
crate boundaries, unstable items may be used within the crate that defines
them.

The `issue` field specifies the associated GitHub [issue number]. This field is
required and all unstable features should have an associated tracking issue. In
rare cases where there is no sensible value `issue = "none"` is used.

The `unstable` attribute infects all sub-items, where the attribute doesn't
have to be reapplied. So if you apply this to a module, all items in the module
will be unstable.

You can make specific sub-items stable by using the `#[stable]` attribute on
them. The stability scheme works similarly to how `pub` works. You can have
public functions of nonpublic modules and you can have stable functions in
unstable modules or vice versa.

Note, however, that due to a [rustc bug], stable items inside unstable modules
*are* available to stable code in that location!  So, for example, stable code
can import `core::intrinsics::transmute` even though `intrinsics` is an
unstable module.  Thus, this kind of nesting should be avoided when possible.

The `unstable` attribute may also have the `soft` value, which makes it a
future-incompatible deny-by-default lint instead of a hard error. This is used
by the `bench` attribute which was accidentally accepted in the past. This
prevents breaking dependencies by leveraging Cargo's lint capping.

[issue number]: https://github.com/rust-lang/rust/issues
[rustc bug]: https://github.com/rust-lang/rust/issues/15702

## stable

The `#[stable(feature = "foo", "since = "1.420.69")]` attribute explicitly
marks an item as stabilized. To do this, follow the instructions in
[Stabilizing Features](./stabilization_guide.md).

Note that stable functions may use unstable things in their body.

## rustc_const_unstable

The `#[rustc_const_unstable(feature = "foo", issue = "1234", reason = "lorem ipsum")]`
has the same interface as the `unstable` attribute. It is used to mark
`const fn` as having their constness be unstable. This allows you to make a
function stable without stabilizing its constness or even just marking an existing
stable function as `const fn` without instantly stabilizing the `const fn`ness.

Furthermore this attribute is needed to mark an intrinsic as `const fn`, because
there's no way to add `const` to functions in `extern` blocks for now.

## rustc_const_stable

The `#[stable(feature = "foo", "since = "1.420.69")]` attribute explicitly marks
a `const fn` as having its constness be `stable`. This attribute can make sense
even on an `unstable` function, if that function is called from another
`rustc_const_stable` function.

Furthermore this attribute is needed to mark an intrinsic as callable from
`rustc_const_stable` functions.


## allow_internal_unstable

Macros, compiler desugarings and `const fn`s expose their bodies to the call
site. To work around not being able to use unstable things in the standard
library's macros, there's the `#[allow_internal_unstable(feature1, feature2)]`
attribute that whitelists the given features for usage in stable macros or
`const fn`s.

Note that `const fn`s are even more special in this regard. You can't just
whitelist any feature, the features need an implementation in
`qualify_min_const_fn.rs`. For example the `const_fn_union` feature gate allows
accessing fields of unions inside stable `const fn`s. The rules for when it's
ok to use such a feature gate are that behavior matches the runtime behavior of
the same code (see also [this blog post][blog]). This means that you may not
create a `const fn` that e.g. transmutes a memory address to an integer,
because the addresses of things are nondeterministic and often unknown at
compile-time.

Always ping @oli-obk, @RalfJung, and @Centril if you are adding more
`allow_internal_unstable` attributes to any `const fn`

## staged_api

Any crate that uses the `stable`, `unstable`, or `rustc_deprecated` attributes
must include the `#![feature(staged_api)]` attribute on the crate.

## rustc_deprecated

The deprecation system shares the same infrastructure as the stable/unstable
attributes. The `rustc_deprecated` attribute is similar to the [`deprecated`
attribute]. It was previously called `deprecated`, but was split off when
`deprecated` was stabilized. The `deprecated` attribute cannot be used in a
`staged_api` crate, `rustc_deprecated` must be used instead. The deprecated
item must also have a `stable` or `unstable` attribute.

`rustc_deprecated` has the following form:

```rust,ignore
#[rustc_deprecated(
    since = "1.38.0",
    reason = "explanation for deprecation",
    suggestion = "other_function"
)]
```

The `suggestion` field is optional. If given, it should be a string that can be
used as a machine-applicable suggestion to correct the warning. This is
typically used when the identifier is renamed, but no other significant changes
are necessary.

Another difference from the `deprecated` attribute is that the `since` field is
actually checked against the current version of `rustc`. If `since` is in a
future version, then the `deprecated_in_future` lint is triggered which is
default `allow`, but most of the standard library raises it to a warning with
`#![warn(deprecated_in_future)]`.

[`deprecated` attribute]: https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-deprecated-attribute

## -Zforce-unstable-if-unmarked

The `-Zforce-unstable-if-unmarked` flag has a variety of purposes to help
enforce that the correct crates are marked as unstable. It was introduced
primarily to allow rustc and the standard library to link to arbitrary crates
on crates.io which do not themselves use `staged_api`. `rustc` also relies on
this flag to mark all of its crates as unstable with the `rustc_private`
feature so that each crate does not need to be carefully marked with
`unstable`.

This flag is automatically applied to all of `rustc` and the standard library
by the bootstrap scripts. This is needed because the compiler and all of its
dependencies are shipped in the sysroot to all users.

This flag has the following effects:

- Marks the crate as "unstable" with the `rustc_private` feature if it is not
  itself marked as stable or unstable.
- Allows these crates to access other forced-unstable crates without any need
  for attributes. Normally a crate would need a `#![feature(rustc_private)]`
  attribute to use other unstable crates. However, that would make it
  impossible for a crate from crates.io to access its own dependencies since
  that crate won't have a `feature(rustc_private)` attribute, but *everything*
  is compiled with `-Zforce-unstable-if-unmarked`.

Code which does not use `-Zforce-unstable-if-unmarked` should include the
`#![feature(rustc_private)]` crate attribute to access these force-unstable
crates. This is needed for things that link `rustc`, such as `miri`, `rls`, or
`clippy`.

[blog]: https://www.ralfj.de/blog/2018/07/19/const.html
