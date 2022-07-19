-passes-previously-accepted =
    this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

-passes-see-issue =
    see issue #{$issue} <https://github.com/rust-lang/rust/issues/{$issue}> for more information

passes-outer-crate-level-attr =
    crate-level attribute should be an inner attribute: add an exclamation mark: `#![foo]`

passes-inner-crate-level-attr =
    crate-level attribute should be in the root module

passes-ignored-attr-with-macro = `#[{$sym}]` is ignored on struct fields, match arms and macro defs
    .warn = {-passes-previously-accepted}
    .note = {-passes-see-issue(issue: "80564")}

passes-ignored-attr = `#[{$sym}]` is ignored on struct fields and match arms
    .warn = {-passes-previously-accepted}
    .note = {-passes-see-issue(issue: "80564")}

passes-inline-ignored-function-prototype = `#[inline]` is ignored on function prototypes

passes-inline-ignored-constants = `#[inline]` is ignored on constants
    .warn = {-passes-previously-accepted}
    .note = {-passes-see-issue(issue: "65833")}

passes-inline-not-fn-or-closure = attribute should be applied to function or closure
    .label = not a function or closure

passes-no-coverage-ignored-function-prototype = `#[no_coverage]` is ignored on function prototypes

passes-no-coverage-propagate =
    `#[no_coverage]` does not propagate into items and must be applied to the contained functions directly

passes-no-coverage-fn-defn = `#[no_coverage]` may only be applied to function definitions

passes-no-coverage-not-coverable = `#[no_coverage]` must be applied to coverable code
    .label = not coverable code

passes-should-be-applied-to-fn = attribute should be applied to a function definition
    .label = not a function definition

passes-naked-tracked-caller = cannot use `#[track_caller]` with `#[naked]`

passes-should-be-applied-to-struct-enum = attribute should be applied to a struct or enum
    .label = not a struct or enum

passes-should-be-applied-to-trait = attribute should be applied to a trait
    .label = not a trait

passes-target-feature-on-statement = {passes-should-be-applied-to-fn}
    .warn = {-passes-previously-accepted}
    .label = {passes-should-be-applied-to-fn.label}

passes-should-be-applied-to-static = attribute should be applied to a static
    .label = not a static

passes-doc-expect-str = doc {$attr_name} attribute expects a string: #[doc({$attr_name} = "a")]

passes-doc-alias-empty = {$attr_str} attribute cannot have empty value

passes-doc-alias-bad-char = {$char_} character isn't allowed in {$attr_str}

passes-doc-alias-start-end = {$attr_str} cannot start or end with ' '

passes-doc-alias-bad-location = {$attr_str} isn't allowed on {$location}

passes-doc-alias-not-an-alias = {$attr_str} is the same as the item's name

passes-doc-alias-duplicated = doc alias is duplicated
    .label = first defined here

passes-doc-alias-not-string-literal = `#[doc(alias("a"))]` expects string literals

passes-doc-alias-malformed =
    doc alias attribute expects a string `#[doc(alias = "a")]` or a list of strings `#[doc(alias("a", "b"))]`

passes-doc-keyword-empty-mod = `#[doc(keyword = "...")]` should be used on empty modules

passes-doc-keyword-not-mod = `#[doc(keyword = "...")]` should be used on modules

passes-doc-keyword-invalid-ident = `{$doc_keyword}` is not a valid identifier

passes-doc-tuple-variadic-not-first =
    `#[doc(tuple_variadic)]` must be used on the first of a set of tuple trait impls with varying arity

passes-doc-keyword-only-impl = `#[doc(keyword = "...")]` should be used on impl blocks

passes-doc-inline-conflict-first = this attribute...
passes-doc-inline-conflict-second = ...conflicts with this attribute
passes-doc-inline-conflict = conflicting doc inlining attributes
    .help = remove one of the conflicting attributes

passes-doc-inline-only-use = this attribute can only be applied to a `use` item
    .label = only applicable on `use` items
    .not-a-use-item-label = not a `use` item
    .note = read <https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#inline-and-no_inline> for more information

passes-doc-attr-not-crate-level =
    `#![doc({$attr_name} = "...")]` isn't allowed as a crate-level attribute

passes-attr-crate-level = this attribute can only be applied at the crate level
    .suggestion = to apply to the crate, use an inner attribute
    .help = to apply to the crate, use an inner attribute
    .note = read <https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#at-the-crate-level> for more information

passes-doc-test-unknown = unknown `doc(test)` attribute `{$path}`

passes-doc-test-takes-list = `#[doc(test(...)]` takes a list of attributes

passes-doc-primitive = `doc(primitive)` should never have been stable

passes-doc-test-unknown-any = unknown `doc` attribute `{$path}`

passes-doc-test-unknown-spotlight = unknown `doc` attribute `{$path}`
    .note = `doc(spotlight)` was renamed to `doc(notable_trait)`
    .suggestion = use `notable_trait` instead
    .no-op-note = `doc(spotlight)` is now a no-op

passes-doc-test-unknown-include = unknown `doc` attribute `{$path}`
    .suggestion = use `doc = include_str!` instead

passes-doc-invalid = invalid `doc` attribute

passes-pass-by-value = `pass_by_value` attribute should be applied to a struct, enum or type alias
    .label = is not a struct, enum or type alias

passes-allow-incoherent-impl =
    `rustc_allow_incoherent_impl` attribute should be applied to impl items.
    .label = the only currently supported targets are inherent methods

passes-has-incoherent-inherent-impl =
    `rustc_has_incoherent_inherent_impls` attribute should be applied to types or traits.
    .label = only adts, extern types and traits are supported

passes-must-use-async =
    `must_use` attribute on `async` functions applies to the anonymous `Future` returned by the function, not the value within
    .label = this attribute does nothing, the `Future`s returned by async functions are already `must_use`

passes-must-use-no-effect = `#[must_use]` has no effect when applied to {$article} {$target}

passes-must-not-suspend = `must_not_suspend` attribute should be applied to a struct, enum, or trait
    .label = is not a struct, enum, or trait

passes-cold = {passes-should-be-applied-to-fn}
    .warn = {-passes-previously-accepted}
    .label = {passes-should-be-applied-to-fn.label}

passes-link = attribute should be applied to an `extern` block with non-Rust ABI
    .warn = {-passes-previously-accepted}
    .label = not an `extern` block
