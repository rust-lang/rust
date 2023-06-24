# Stability attributes

This section is about the stability attributes and schemes that allow stable
APIs to use unstable APIs internally in the rustc standard library.

**NOTE**: this section is for *library* features, not *language* features. For instructions on
stabilizing a language feature see [Stabilizing Features](./stabilization_guide.md).

<!-- toc -->

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
The `#[stable(feature = "foo", since = "1.420.69")]` attribute explicitly
marks an item as stabilized. Note that stable functions may use unstable things in their body.

## rustc_const_unstable

The `#[rustc_const_unstable(feature = "foo", issue = "1234", reason = "lorem ipsum")]`
has the same interface as the `unstable` attribute. It is used to mark
`const fn` as having their constness be unstable. This allows you to make a
function stable without stabilizing its constness or even just marking an existing
stable function as `const fn` without instantly stabilizing the `const fn`ness.

Furthermore this attribute is needed to mark an intrinsic as `const fn`, because
there's no way to add `const` to functions in `extern` blocks for now.

## rustc_const_stable

The `#[rustc_const_stable(feature = "foo", since = "1.420.69")]` attribute explicitly marks
a `const fn` as having its constness be `stable`. This attribute can make sense
even on an `unstable` function, if that function is called from another
`rustc_const_stable` function.

Furthermore this attribute is needed to mark an intrinsic as callable from
`rustc_const_stable` functions.

## Stabilizing a library feature

To stabilize a feature, follow these steps:

1. Ask a **@T-libs-api** member to start an FCP on the tracking issue and wait for
   the FCP to complete (with `disposition-merge`).
2. Change `#[unstable(...)]` to `#[stable(since = "CURRENT_RUSTC_VERSION")]`.
3. Remove `#![feature(...)]` from any test or doc-test for this API. If the feature is used in the
   compiler or tools, remove it from there as well.
4. If applicable, change `#[rustc_const_unstable(...)]` to
   `#[rustc_const_stable(since = "CURRENT_RUSTC_VERSION")]`.
5. Open a PR against `rust-lang/rust`.
   - Add the appropriate labels: `@rustbot modify labels: +T-libs-api`.
   - Link to the tracking issue and say "Closes #XXXXX".

You can see an example of stabilizing a feature with
[tracking issue #81656 with FCP](https://github.com/rust-lang/rust/issues/81656)
and the associated
[implementation PR #84642](https://github.com/rust-lang/rust/pull/84642).

## allow_internal_unstable

Macros and compiler desugarings expose their bodies to the call
site. To work around not being able to use unstable things in the standard
library's macros, there's the `#[allow_internal_unstable(feature1, feature2)]`
attribute that allows the given features to be used in stable macros.

## rustc_allow_const_fn_unstable

`const fn`, while not directly exposing their body to the world, are going to get
evaluated at compile time in stable crates. If their body does something const-unstable,
that could lock us into certain features indefinitely by accident. Thus no unstable const
features are allowed inside stable `const fn`.

However, sometimes we do know that a feature will get
stabilized, just not when, or there is a stable (but e.g. runtime-slow) workaround, so we
could always fall back to some stable version if we scrapped the unstable feature.
In those cases, the rustc_allow_const_fn_unstable attribute can be used to allow some
unstable features in the body of a stable `const fn`.

You also need to take care to uphold the `const fn` invariant that calling it at runtime and
compile-time needs to behave the same (see also [this blog post][blog]). This means that you
may not create a `const fn` that e.g. transmutes a memory address to an integer,
because the addresses of things are nondeterministic and often unknown at
compile-time.

Always ping @rust-lang/wg-const-eval if you are adding more
`rustc_allow_const_fn_unstable` attributes to any `const fn`.

## staged_api

Any crate that uses the `stable` or `unstable` attributes must include the
`#![feature(staged_api)]` attribute on the crate.

## deprecated

Deprecations in the standard library are nearly identical to deprecations in
user code. When `#[deprecated]` is used on an item, it must also have a `stable`
or `unstable `attribute.

`deprecated` has the following form:

```rust,ignore
#[deprecated(
    since = "1.38.0",
    note = "explanation for deprecation",
    suggestion = "other_function"
)]
```

The `suggestion` field is optional. If given, it should be a string that can be
used as a machine-applicable suggestion to correct the warning. This is
typically used when the identifier is renamed, but no other significant changes
are necessary. When the `suggestion` field is used, you need to have
`#![feature(deprecated_suggestion)]` at the crate root.

Another difference from user code is that the `since` field is actually checked
against the current version of `rustc`. If `since` is in a future version, then
the `deprecated_in_future` lint is triggered which is default `allow`, but most
of the standard library raises it to a warning with
`#![warn(deprecated_in_future)]`.

[blog]: https://www.ralfj.de/blog/2018/07/19/const.html
