# Stability attributes

This section is about the stability attributes and schemes that allow stable
APIs to use unstable APIs internally in the rustc standard library.

**NOTE**: this section is for *library* features, not *language* features. For instructions on
stabilizing a language feature see [Stabilizing Features](./stabilization_guide.md).

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

Previously, due to a [rustc bug], stable items inside unstable modules were
available to stable code in that location.
As of <!-- date-check --> September 2024, items with [accidentally stabilized
paths] are marked with the `#[rustc_allowed_through_unstable_modules]` attribute
to prevent code dependent on those paths from breaking. Do *not* add this attribute
to any more items unless that is needed to avoid breaking changes.

The `unstable` attribute may also have the `soft` value, which makes it a
future-incompatible deny-by-default lint instead of a hard error. This is used
by the `bench` attribute which was accidentally accepted in the past. This
prevents breaking dependencies by leveraging Cargo's lint capping.

[issue number]: https://github.com/rust-lang/rust/issues
[rustc bug]: https://github.com/rust-lang/rust/issues/15702
[accidentally stabilized paths]: https://github.com/rust-lang/rust/issues/113387

## stable
The `#[stable(feature = "foo", since = "1.420.69")]` attribute explicitly
marks an item as stabilized. Note that stable functions may use unstable things in their body.

## rustc_const_unstable

The `#[rustc_const_unstable(feature = "foo", issue = "1234", reason = "lorem
ipsum")]` has the same interface as the `unstable` attribute. It is used to mark
`const fn` as having their constness be unstable. This is only needed in rare cases:
- If a `const fn` makes use of unstable language features or intrinsics.
  (The compiler will tell you to add the attribute if you run into this.)
- If a `const fn` is `#[stable]` but not yet intended to be const-stable.
- To change the feature gate that is required to call a const-unstable intrinsic.

Const-stability differs from regular stability in that it is *recursive*: a
`#[rustc_const_unstable(...)]` function cannot even be indirectly called from stable code. This is
to avoid accidentally leaking unstable compiler implementation artifacts to stable code or locking
us into the accidental quirks of an incomplete implementation. See the rustc_const_stable_indirect
and rustc_allow_const_fn_unstable attributes below for how to fine-tune this check.

## rustc_const_stable

The `#[rustc_const_stable(feature = "foo", since = "1.420.69")]` attribute explicitly marks
a `const fn` as having its constness be `stable`.

## rustc_const_stable_indirect

The `#[rustc_const_stable_indirect]` attribute can be added to a `#[rustc_const_unstable(...)]`
function to make it callable from `#[rustc_const_stable(...)]` functions. This indicates that the
function is ready for stable in terms of its implementation (i.e., it doesn't use any unstable
compiler features); the only reason it is not const-stable yet are API concerns.

This should also be added to lang items for which const-calls are synthesized in the compiler, to
ensure those calls do not bypass recursive const stability rules.

## rustc_intrinsic_const_stable_indirect

On an intrinsic, this attribute marks the intrinsic as "ready to be used by public stable functions".
If the intrinsic has a `rustc_const_unstable` attribute, it should be removed.
**Adding this attribute to an intrinsic requires t-lang and wg-const-eval approval!**

## rustc_default_body_unstable

The `#[rustc_default_body_unstable(feature = "foo", issue = "1234", reason =
"lorem ipsum")]` attribute has the same interface as the `unstable` attribute.
It is used to mark the default implementation for an item within a trait as
unstable.
A trait with a default-body-unstable item can be implemented stably by providing
an explicit body for any such item, or the default body can be used by enabling
its corresponding `#![feature]`.

## Stabilizing a library feature

To stabilize a feature, follow these steps:

1. Ask a **@T-libs-api** member to start an FCP on the tracking issue and wait for
   the FCP to complete (with `disposition-merge`).
2. Change `#[unstable(...)]` to `#[stable(since = "CURRENT_RUSTC_VERSION")]`.
3. Remove `#![feature(...)]` from any test or doc-test for this API. If the feature is used in the
   compiler or tools, remove it from there as well.
4. If this is a `const fn`, add `#[rustc_const_stable(since = "CURRENT_RUSTC_VERSION")]`.
   Alternatively, if this is not supposed to be const-stabilized yet,
   add `#[rustc_const_unstable(...)]` for some new feature gate (with a new tracking issue).
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

Note that if a macro is used in const context and generates a call to a
`#[rustc_const_unstable(...)]` function, that will *still* be rejected even with
`allow_internal_unstable`. Add `#[rustc_const_stable_indirect]` to the function to ensure the macro
cannot accidentally bypass the recursive const stability checks.

## rustc_allow_const_fn_unstable

As explained above, no unstable const features are allowed inside stable `const fn`, not even
indirectly.

However, sometimes we do know that a feature will get stabilized, just not when, or there is a
stable (but e.g. runtime-slow) workaround, so we could always fall back to some stable version if we
scrapped the unstable feature. In those cases, the `[rustc_allow_const_fn_unstable(feature1,
feature2)]` attribute can be used to allow some unstable features in the body of a stable (or
indirectly stable) `const fn`.

You also need to take care to uphold the `const fn` invariant that calling it at runtime and
compile-time needs to behave the same (see also [this blog post][blog]). This means that you
may not create a `const fn` that e.g. transmutes a memory address to an integer,
because the addresses of things are nondeterministic and often unknown at
compile-time.

**Always ping @rust-lang/wg-const-eval if you are adding more
`rustc_allow_const_fn_unstable` attributes to any `const fn`.**

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

## unstable_feature_bound
The `#[unstable_feature_bound(foo)]` attribute can be used together with `#[unstable]` attribute to mark an `impl` of stable type and stable trait as unstable. In std/core, an item annotated with `#[unstable_feature_bound(foo)]` can only be used by another item that is also annotated with `#[unstable_feature_bound(foo)]`. Outside of std/core, using an item with `#[unstable_feature_bound(foo)]` requires the feature to be enabled with `#![feature(foo)]` attribute on the crate.

Currently, the items that can be annotated with `#[unstable_feature_bound]` are:
- `impl`
- free function
- trait

[blog]: https://www.ralfj.de/blog/2018/07/19/const.html
