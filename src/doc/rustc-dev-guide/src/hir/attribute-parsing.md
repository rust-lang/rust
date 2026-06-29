# Attribute Parsing

Attributes come in two types: *inert* (or *built-in*) and *active* (*non-builtin*).
Inert attributes are parsed during AST lowering to the HIR, while active attributes are expanded during expansion.
For more information about the difference, see [the page about attributes][attributes_page].

During [AST lowering][lowering], inert attributes are converted from an unparsed `TokenStream` to a parsed representation.
The parsed form [is defined][hir_attrs] in the `rustc_hir` crate, and the parsers [are defined][attr_parsing] in the `rustc_attr_parsing` crate.

## A step by step guide of adding a new inert attribute parser

1. Add the attribute name to the [BUILTIN_ATTRIBUTES].
   This list defines the set of inert attributes.
2. Add a variant to `AttributeKind` in the [hir definition of attributes][hir_attrs].
   This will define the parsed form of the attribute that the attribute parser will produce.
3. Add your variant to the match in `rustc_hir/attrs/encode_cross_crate.rs`, which should return whether your attribute should be visible in dependent crates.
   This is usually `No` for codegen related attributes, and `Yes` for analysis related attributes.
4. Create a new struct in `rustc_attr_parsing/attributes/*.rs` that will hold the state for your attribute parser.
   For most parsers, this will be an empty struct.
5. Implement one of the attribute parsing traits for this struct:
   * `NoArgsAttributeParser` for attributes that may appear only a single time per item, and take no arguments.
     For example `#[no_mangle]`.
   * `SingleAttributeParser` for attributes that may appear only a single time per item, and may take arguments.
     For example `#[inline(...)]`.
   * `CombineAttributeParser` for attributes that may appear multiple times per item, and whose occurrences are combined into a single parsed attribute.
     For example `#[feature(...)]`.
   * `AttributeParser` for attributes that don't fit cleanly into one of the above traits.
     For example when multiple different attributes need to be combined into a parsed attribute (such as `#[stable]` and `#[unstable]`),
6. Implement the associated constants of the attribute parser trait that you chose.
   Not all of these constants are applicable to all traits.
   * `PATH`: The name of the attribute that is added
   * `TEMPLATE`: A structural description of your attribute, used only to guide diagnostics.
     Use the [template!] macro to construct this type.
   * `STABILITY`: Whether your attribute is stable (ungated) or unstable (feature gated).
   * `ALLOWED_TARGETS`: Which targets the attribute may appear on.
     For new attributes, all incorrect targets should error, so use the `AllowedTargets::AllowList` for new attributes, and only use `Policy::Allow` to allow targets.
   * `ON_DUPLICATE`: Controls the behavior when the attribute incorrectly appears multiple times.
     Should usually be `OnDuplicate::Error` for new attributes, this is the default.
   * `SAFETY`: Controls whether using the attribute requires marking it as unsafe, such as `#[unsafe(no_mangle)]`.
     Should usually be `AttributeSafety::Normal`, this is the default.
7. Implement the associated parsing functions.
   In general, you will be provided with an `AcceptContext` providing the context in which the attribute is used, and `ArgParser` which contains the arguments that were provided to the attribute.
   When the arguments are incorrect, you should emit an error using the suite of `cx.adcx().expected_*` function and then `return None`.
   Sometimes these two steps can be combined, using the `cx.expect_*` suite of functions, on which you can use the `?` operator to early-return.
   The attribute parser should do all checks that are possible to ensure the attribute is valid, before returning the parsed attribute.
8. Register the attribute parser by adding it to [the list of attribute parsers][attribute_parsers].
9. If not all validation could be done in the attribute parser because not enough information was available, add an extra validation step in `rustc_passes/check_attr.rs`.
   This code is ran right after the HIR is fully built, so it has access to the full HIR for analysis.

## Using the attribute parsers

There are two ways of using an attribute parser, referred to as `Early` and `Late` attribute parsing.

### Late (the usual way)
To use the parsed representation after the HIR is built, use the `find_attr!` macro [defined here][find_attr].
This macro can be used in the following ways:
* `find_attr!(tcx, <def_id>, Variant(...))` to find an attribute on a def id.
* `find_attr!(tcx, <hir_id>, Variant(...))` to find an attribute on a HIR id.
* `find_attr!(tcx, crate, Variant(...))` to find an attribute on the current crate.
* `find_attr!(attrs, Variant(...))` to find an attribute in attrs, a `&[hir::Attribute]`.

`Variant` is a pattern matching one of the `AttributeKind` variants, and can take one of the following shapes:
* `find_attr!(..., Variant)` will return a boolean representing whether the attribute is present.
* `find_attr!(..., Variant(a, b, _) => (a, b))` will return the value after the `=>`, constructed from fields bound by the pattern.

### Early (also referred to as "limited")
To use the parsed representation before the HIR is built, you usually need to use the [`AttributeParser::parse_limited`][parse_limited] function.
This function takes a slice of `&[ast::Attribute]`s, and the name of an attribute you're interested in, and parses the attribute if it is present.

No diagnostics will be emitted when parsing limited.
Lints are not emitted at all, while errors will be emitted as a delayed bugs.
In other words, we expect attributes parsed with `parse_limited` to be reparsed later during ast lowering where we do emit the errors.


[attributes_page]: ../attributes.md
[lowering]: ./lowering.md
[hir_attrs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/attrs/index.html
[attr_parsing]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_attr_parsing/
[attribute_parsers]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_attr_parsing/context/static.ATTRIBUTE_PARSERS.html
[builtin_attributes]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_feature/builtin_attrs/static.BUILTIN_ATTRIBUTES.html
[template!]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_attr_parsing/macro.template.html
[find_attr]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/macro.find_attr.html
[parse_limited]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_attr_parsing/interface/struct.AttributeParser.html#method.parse_limited
