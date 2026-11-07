# rustdoc HTML test suite categories

This is a high-level summary of the organization of the rustdoc HTML test suite
(`tests/rustdoc-html`). It is not intended to be *prescriptive*, but instead provide a quick survey
of existing groupings.

For now, only immediate subdirectories under `tests/rustdoc-html` are described.

## `tests/rustdoc-html/anchors`

Check for the link anchors. Rustdoc uses anchors when we want to point an element on an item
page, like the method of a type.

## `tests/rustdoc-html/assoc`

Contains tests checking how an associated item like a trait method is rendered in the
docs.

## `tests/rustdoc-html/async`

Contains tests checking the rendering of `async` items.

## `tests/rustdoc-html/auto`

Contains tests related to auto traits implementations.

## `tests/rustdoc-html/codeblock`

Contains tests to check the rendering of the code blocks (also called code examples).

## `tests/rustdoc-html/const-generics`

Mostly the same as `tests/rustdoc-html/constant` except that this folder focuses on the
rendering of const generics. Same problematics overall.

## `tests/rustdoc-html/constant`

Contains tests to check constant items rendering. Constant items can sometimes have
code that needs to be kept as is or inferred, making this a very tricky topic in rustdoc. :)

## `tests/rustdoc-html/cross-crate-info`

Contains tests to check how information is used across crates. It only contains folders
which all contain an `auxiliary` subfolder for their own dependencies (and avoid potential naming
conflicts, in particular for simpler crate names).

## `tests/rustdoc-html/deref`

Contains tests to check the display of items implementing the `Deref` and/or `DerefMut` traits.

## `tests/rustdoc-html/doc-cfg`

Contains tests to check the rustdoc `doc_cfg` feature.

## `tests/rustdoc-html/enum`

Contains tests for the rendering of enum types and in particular its variants.

## `tests/rustdoc-html/extern`

Contains tests to check how extern items (like `extern "C"` functions) are rendered.

## `tests/rustdoc-html/footnote`

Contains tests checking the markdown footnote features and how it is rendered.

## `tests/rustdoc-html/generic-associated-types`

Contains tests checking the rendering of items when they have generic associated types,
like `Iterator::Item` where `Item` would instead be `Item<'a> where Self: 'a`. Fun times. :)

## `tests/rustdoc-html/hidden`

Contains tests checking both `doc(hidden)` and the `--document-hidden-items` CLI flag.

## `tests/rustdoc-html/impl`

Contains tests checking the rendering of impl blocks. More general than
`tests/rustdoc-html/generic-associated-types`.

## `tests/rustdoc-html/inline_cross`

Contains tests which check inlined foreign items and how they render based on the limitless
list of corner cases we have to handle with reexports.

## `tests/rustdoc-html/inline_local`

Similar to `tests/rustdoc-html/inline_cross` except this time it's reexports of local items.
Still a lot of "fun" and dark magic.

## `tests/rustdoc-html/intra-doc`

Contains tests which check the intra doc links feature. This feature is very important in
rustdoc as it allows to simply write the name or the path of an item to get its link
generated. It comes with a lot of corner cases around scope and namespaces (illustrated
by the very high number of tests in this folder).

## `tests/rustdoc-html/jump-to-def`

This folder contains tests for the "jump to def(inition)" feature which is enabled with the
`--generate-link-to-definition` CLI flag.

## `tests/rustdoc-html/macro`

Contains tests checking the rendering of macro items (both `derive` and `macro_rules!`).

## `tests/rustdoc-html/macro-expansion`

Contains tests checking the `--generate-macro-expansion` CLI flag.

## `tests/rustdoc-html/notable-trait`

Contains tests checking the `doc(notable_trait)` attribute.

## `tests/rustdoc-html/primitive`

Everything related to primitive types, like intra-doc links pointing to them, documenting
them with the `rustc_doc_primitive` attribute, reexporting them, etc.

## `tests/rustdoc-html/private`

Everything related to non-`doc(hidden)` items which aren't public. Either it's struct/union
fields, impl blocks containing only private items or the handling of the
`--document-private-items` CLI flag.

## `tests/rustdoc-html/reexport`

Another folder about reexports (might be worth merging the three of them?). It checks
everything related to reexports like concatenated docs and attributes, inlining, transitive
reexported crates, etc.

## `tests/rustdoc-html/sidebar`

Contains tests checking the content of the sidebar like sections or links to items present on the
page, etc.

## `tests/rustdoc-html/source-code-pages`

Contains tests checking the content of the source code pages (when you click on the `source`
links).

## `tests/rustdoc-html/synthetic_auto`

Somewhat equivalent to the `auto` folder's tests. Both should likely be merged.

## `tests/rustdoc-html/typedef`

Contains tests checking type aliases (`type X = Y`) rendering. In particular when the
alias contains generics, meaning a specialized aliased type, allowing in some cases
more auto traits implementations.

## `tests/rustdoc-html/union`

Same as `tests/rustdoc-html/enum` but for union items.
