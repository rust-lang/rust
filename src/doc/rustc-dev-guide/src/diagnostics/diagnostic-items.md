# Diagnostic Items

While writing lints it's common to check for specific types, traits and
functions. This raises the question on how to check for these. Types can be
checked by their complete type path. However, this requires hard coding paths
and can lead to misclassifications in some edge cases. To counteract this,
rustc has introduced diagnostic items that are used to identify types via
[`Symbol`]s.

## Finding diagnostic items

Diagnostic items are added to items inside `rustc`/`std`/`core`/`alloc` with the
`rustc_diagnostic_item` attribute. The item for a specific type can be found by
opening the source code in the documentation and looking for this attribute.
Note that it's often added with the `cfg_attr` attribute to avoid compilation
errors during tests. A definition often looks like this:

```rs
// This is the diagnostic item for this type   vvvvvvv
#[cfg_attr(not(test), rustc_diagnostic_item = "Penguin")]
struct Penguin;
```

Diagnostic items are usually only added to traits,
types,
and standalone functions.
If the goal is to check for an associated type or method,
please use the diagnostic item of the item and reference
[*Using Diagnostic Items*](#using-diagnostic-items).

## Adding diagnostic items

A new diagnostic item can be added with these two steps:

1. Find the target item inside the Rust repo. Now add the diagnostic item as a
   string via the `rustc_diagnostic_item` attribute. This can sometimes cause
   compilation errors while running tests. These errors can be avoided by using
   the `cfg_attr` attribute with the `not(test)` condition (it's fine adding
   then for all `rustc_diagnostic_item` attributes as a preventive manner). At
   the end, it should look like this:

    ```rs
    // This will be the new diagnostic item        vvv
    #[cfg_attr(not(test), rustc_diagnostic_item = "Cat")]
    struct Cat;
    ```

    For the naming conventions of diagnostic items, please refer to
    [*Naming Conventions*](#naming-conventions).

2. <!-- date-check: Feb 2023 -->
   Diagnostic items in code are accessed via symbols in
   [`rustc_span::symbol::sym`].
   To add your newly-created diagnostic item,
   simply open the module file,
   and add the name (In this case `Cat`) at the correct point in the list.

Now you can create a pull request with your changes. :tada:

> NOTE:
> When using diagnostic items in other projects like Clippy,
> it might take some time until the repos get synchronized.

## Naming conventions

Diagnostic items don't have a naming convention yet.
Following are some guidelines that should be used in future,
but might differ from existing names:

* Types, traits, and enums are named using UpperCamelCase
  (Examples: `Iterator` and `HashMap`)
* For type names that are used multiple times,
  like `Writer`,
  it's good to choose a more precise name,
  maybe by adding the module to it
  (Example: `IoWriter`)
* Associated items should not get their own diagnostic items,
  but instead be accessed indirectly by the diagnostic item
  of the type they're originating from.
* Freestanding functions like `std::mem::swap()` should be named using
  `snake_case` with one important (export) module as a prefix
  (Examples: `mem_swap` and `cmp_max`)
* Modules should usually not have a diagnostic item attached to them.
  Diagnostic items were added to avoid the usage of paths,
  and using them on modules would therefore most likely be counterproductive.

## Using diagnostic items

In rustc, diagnostic items are looked up via [`Symbol`]s from inside the
[`rustc_span::symbol::sym`] module. These can then be mapped to [`DefId`]s
using [`TyCtxt::get_diagnostic_item()`] or checked if they match a [`DefId`]
using [`TyCtxt::is_diagnostic_item()`]. When mapping from a diagnostic item to
a [`DefId`], the method will return a `Option<DefId>`. This can be `None` if
either the symbol isn't a diagnostic item or the type is not registered, for
instance when compiling with `#[no_std]`.
All the following examples are based on [`DefId`]s and their usage.

### Example: Checking for a type

```rust
use rustc_span::symbol::sym;

/// This example checks if the given type (`ty`) has the type `HashMap` using
/// `TyCtxt::is_diagnostic_item()`
fn example_1(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    match ty.kind() {
        ty::Adt(adt, _) => cx.tcx.is_diagnostic_item(sym::HashMap, adt.did()),
        _ => false,
    }
}
```

### Example: Checking for a trait implementation

```rust
/// This example checks if a given [`DefId`] from a method is part of a trait
/// implementation defined by a diagnostic item.
fn is_diag_trait_item(
    cx: &LateContext<'_>,
    def_id: DefId,
    diag_item: Symbol
) -> bool {
    if let Some(trait_did) = cx.tcx.trait_of_item(def_id) {
        return cx.tcx.is_diagnostic_item(diag_item, trait_did);
    }
    false
}
```

### Associated Types

Associated types of diagnostic items can be accessed indirectly by first
getting the [`DefId`] of the trait and then calling
[`TyCtxt::associated_items()`]. This returns an [`AssocItems`] object which can
be used for further checks. Checkout
[`clippy_utils::ty::get_iterator_item_ty()`] for an example usage of this.

### Usage in Clippy

Clippy tries to use diagnostic items where possible and has developed some
wrapper and utility functions. Please also refer to its documentation when
using diagnostic items in Clippy. (See [*Common tools for writing
lints*][clippy-Common-tools-for-writing-lints].)

## Related issues

These are probably only interesting to people
who really want to take a deep dive into the topic :)

* [rust#60966]: The Rust PR that introduced diagnostic items
* [rust-clippy#5393]: Clippy's tracking issue for moving away from hard coded paths to
  diagnostic item

<!-- Links -->

[`rustc_span::symbol::sym`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/symbol/sym/index.html
[`Symbol`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/symbol/struct.Symbol.html
[`DefId`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.DefId.html
[`TyCtxt::get_diagnostic_item()`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.get_diagnostic_item
[`TyCtxt::is_diagnostic_item()`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.is_diagnostic_item
[`TyCtxt::associated_items()`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.associated_items
[`AssocItems`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/assoc/struct.AssocItems.html
[`clippy_utils::ty::get_iterator_item_ty()`]: https://github.com/rust-lang/rust-clippy/blob/305177342fbc622c0b3cb148467bab4b9524c934/clippy_utils/src/ty.rs#L55-L72
[clippy-Common-tools-for-writing-lints]: https://doc.rust-lang.org/nightly/clippy/development/common_tools_writing_lints.html
[rust#60966]: https://github.com/rust-lang/rust/pull/60966
[rust-clippy#5393]: https://github.com/rust-lang/rust-clippy/issues/5393
