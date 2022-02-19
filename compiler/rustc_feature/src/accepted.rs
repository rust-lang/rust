//! List of the accepted feature gates.

use super::{to_nonzero, Feature, State};
use rustc_span::symbol::sym;

macro_rules! declare_features {
    ($(
        $(#[doc = $doc:tt])* (accepted, $feature:ident, $ver:expr, $issue:expr, None),
    )+) => {
        /// Those language feature has since been Accepted (it was once Active)
        pub const ACCEPTED_FEATURES: &[Feature] = &[
            $(
                Feature {
                    state: State::Accepted,
                    name: sym::$feature,
                    since: $ver,
                    issue: to_nonzero($issue),
                    edition: None,
                }
            ),+
        ];
    }
}

#[rustfmt::skip]
declare_features! (
    // -------------------------------------------------------------------------
    // feature-group-start: for testing purposes
    // -------------------------------------------------------------------------

    /// A temporary feature gate used to enable parser extensions needed
    /// to bootstrap fix for #5723.
    (accepted, issue_5723_bootstrap, "1.0.0", None, None),
    /// These are used to test this portion of the compiler,
    /// they don't actually mean anything.
    (accepted, test_accepted_feature, "1.0.0", None, None),
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!
    // Features are listed in alphabetical order. Tidy will fail if you don't keep it this way.
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!

    // -------------------------------------------------------------------------
    // feature-group-end: for testing purposes
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // feature-group-start: accepted features
    // -------------------------------------------------------------------------

    /// Allows the sysV64 ABI to be specified on all platforms
    /// instead of just the platforms on which it is the C ABI.
    (accepted, abi_sysv64, "1.24.0", Some(36167), None),
    /// Allows the definition of associated constants in `trait` or `impl` blocks.
    (accepted, associated_consts, "1.20.0", Some(29646), None),
    /// Allows using associated `type`s in `trait`s.
    (accepted, associated_types, "1.0.0", None, None),
    /// Allows free and inherent `async fn`s, `async` blocks, and `<expr>.await` expressions.
    (accepted, async_await, "1.39.0", Some(50547), None),
    /// Allows all literals in attribute lists and values of key-value pairs.
    (accepted, attr_literals, "1.30.0", Some(34981), None),
    /// Allows overloading augmented assignment operations like `a += b`.
    (accepted, augmented_assignments, "1.8.0", Some(28235), None),
    /// Allows mixing bind-by-move in patterns and references to those identifiers in guards.
    (accepted, bind_by_move_pattern_guards, "1.39.0", Some(15287), None),
    /// Allows bindings in the subpattern of a binding pattern.
    /// For example, you can write `x @ Some(y)`.
    (accepted, bindings_after_at, "1.56.0", Some(65490), None),
    /// Allows empty structs and enum variants with braces.
    (accepted, braced_empty_structs, "1.8.0", Some(29720), None),
    /// Allows `#[cfg_attr(predicate, multiple, attributes, here)]`.
    (accepted, cfg_attr_multi, "1.33.0", Some(54881), None),
    /// Allows the use of `#[cfg(doctest)]`, set when rustdoc is collecting doctests.
    (accepted, cfg_doctest, "1.40.0", Some(62210), None),
    /// Enables `#[cfg(panic = "...")]` config key.
    (accepted, cfg_panic, "1.60.0", Some(77443), None),
    /// Allows `cfg(target_feature = "...")`.
    (accepted, cfg_target_feature, "1.27.0", Some(29717), None),
    /// Allows `cfg(target_vendor = "...")`.
    (accepted, cfg_target_vendor, "1.33.0", Some(29718), None),
    /// Allows implementing `Clone` for closures where possible (RFC 2132).
    (accepted, clone_closures, "1.26.0", Some(44490), None),
    /// Allows coercing non capturing closures to function pointers.
    (accepted, closure_to_fn_coercion, "1.19.0", Some(39817), None),
    /// Allows usage of the `compile_error!` macro.
    (accepted, compile_error, "1.20.0", Some(40872), None),
    /// Allows `impl Trait` in function return types.
    (accepted, conservative_impl_trait, "1.26.0", Some(34511), None),
    /// Allows calling constructor functions in `const fn`.
    (accepted, const_constructor, "1.40.0", Some(61456), None),
    /// Allows calling `transmute` in const fn
    (accepted, const_fn_transmute, "1.56.0", Some(53605), None),
    /// Allows accessing fields of unions inside `const` functions.
    (accepted, const_fn_union, "1.56.0", Some(51909), None),
    /// Allows unsizing coercions in `const fn`.
    (accepted, const_fn_unsize, "1.54.0", Some(64992), None),
    /// Allows const generics to have default values (e.g. `struct Foo<const N: usize = 3>(...);`).
    (accepted, const_generics_defaults, "1.59.0", Some(44580), None),
    /// Allows the use of `if` and `match` in constants.
    (accepted, const_if_match, "1.46.0", Some(49146), None),
    /// Allows indexing into constant arrays.
    (accepted, const_indexing, "1.26.0", Some(29947), None),
    /// Allows let bindings, assignments and destructuring in `const` functions and constants.
    /// As long as control flow is not implemented in const eval, `&&` and `||` may not be used
    /// at the same time as let bindings.
    (accepted, const_let, "1.33.0", Some(48821), None),
    /// Allows the use of `loop` and `while` in constants.
    (accepted, const_loop, "1.46.0", Some(52000), None),
    /// Allows panicking during const eval (producing compile-time errors).
    (accepted, const_panic, "1.57.0", Some(51999), None),
    /// Allows dereferencing raw pointers during const eval.
    (accepted, const_raw_ptr_deref, "1.58.0", Some(51911), None),
    /// Allows implementing `Copy` for closures where possible (RFC 2132).
    (accepted, copy_closures, "1.26.0", Some(44490), None),
    /// Allows `crate` in paths.
    (accepted, crate_in_paths, "1.30.0", Some(45477), None),
    /// Allows using assigning a default type to type parameters in algebraic data type definitions.
    (accepted, default_type_params, "1.0.0", None, None),
    /// Allows `#[deprecated]` attribute.
    (accepted, deprecated, "1.9.0", Some(29935), None),
    /// Allows the use of destructuring assignments.
    (accepted, destructuring_assignment, "1.59.0", Some(71126), None),
    /// Allows `#[doc(alias = "...")]`.
    (accepted, doc_alias, "1.48.0", Some(50146), None),
    /// Allows `..` in tuple (struct) patterns.
    (accepted, dotdot_in_tuple_patterns, "1.14.0", Some(33627), None),
    /// Allows `..=` in patterns (RFC 1192).
    (accepted, dotdoteq_in_patterns, "1.26.0", Some(28237), None),
    /// Allows `Drop` types in constants (RFC 1440).
    (accepted, drop_types_in_const, "1.22.0", Some(33156), None),
    /// Allows using `dyn Trait` as a syntax for trait objects.
    (accepted, dyn_trait, "1.27.0", Some(44662), None),
    /// Allows integer match exhaustiveness checking (RFC 2591).
    (accepted, exhaustive_integer_patterns, "1.33.0", Some(50907), None),
    /// Allows arbitrary expressions in key-value attributes at parse time.
    (accepted, extended_key_value_attributes, "1.54.0", Some(78835), None),
    /// Allows resolving absolute paths as paths from other crates.
    (accepted, extern_absolute_paths, "1.30.0", Some(44660), None),
    /// Allows `extern crate foo as bar;`. This puts `bar` into extern prelude.
    (accepted, extern_crate_item_prelude, "1.31.0", Some(55599), None),
    /// Allows `extern crate self as foo;`.
    /// This puts local crate root into extern prelude under name `foo`.
    (accepted, extern_crate_self, "1.34.0", Some(56409), None),
    /// Allows access to crate names passed via `--extern` through prelude.
    (accepted, extern_prelude, "1.30.0", Some(44660), None),
    /// Allows field shorthands (`x` meaning `x: x`) in struct literal expressions.
    (accepted, field_init_shorthand, "1.17.0", Some(37340), None),
    /// Allows `#[must_use]` on functions, and introduces must-use operators (RFC 1940).
    (accepted, fn_must_use, "1.27.0", Some(43302), None),
    /// Allows capturing variables in scope using format_args!
    (accepted, format_args_capture, "1.58.0", Some(67984), None),
    /// Allows attributes on lifetime/type formal parameters in generics (RFC 1327).
    (accepted, generic_param_attrs, "1.27.0", Some(48848), None),
    /// Allows the `#[global_allocator]` attribute.
    (accepted, global_allocator, "1.28.0", Some(27389), None),
    // FIXME: explain `globs`.
    (accepted, globs, "1.0.0", None, None),
    /// Allows using the `u128` and `i128` types.
    (accepted, i128_type, "1.26.0", Some(35118), None),
    /// Allows the use of `if let` expressions.
    (accepted, if_let, "1.0.0", None, None),
    /// Allows top level or-patterns (`p | q`) in `if let` and `while let`.
    (accepted, if_while_or_patterns, "1.33.0", Some(48215), None),
    /// Allows lifetime elision in `impl` headers. For example:
    /// + `impl<I:Iterator> Iterator for &mut Iterator`
    /// + `impl Debug for Foo<'_>`
    (accepted, impl_header_lifetime_elision, "1.31.0", Some(15872), None),
    /// Allows using `a..=b` and `..=b` as inclusive range syntaxes.
    (accepted, inclusive_range_syntax, "1.26.0", Some(28237), None),
    /// Allows inferring outlives requirements (RFC 2093).
    (accepted, infer_outlives_requirements, "1.30.0", Some(44493), None),
    /// Allows irrefutable patterns in `if let` and `while let` statements (RFC 2086).
    (accepted, irrefutable_let_patterns, "1.33.0", Some(44495), None),
    /// Allows some increased flexibility in the name resolution rules,
    /// especially around globs and shadowing (RFC 1560).
    (accepted, item_like_imports, "1.15.0", Some(35120), None),
    /// Allows `break {expr}` with a value inside `loop`s.
    (accepted, loop_break_value, "1.19.0", Some(37339), None),
    /// Allows use of `?` as the Kleene "at most one" operator in macros.
    (accepted, macro_at_most_once_rep, "1.32.0", Some(48075), None),
    /// Allows macro attributes to observe output of `#[derive]`.
    (accepted, macro_attributes_in_derive_output, "1.57.0", Some(81119), None),
    /// Allows use of the `:lifetime` macro fragment specifier.
    (accepted, macro_lifetime_matcher, "1.27.0", Some(34303), None),
    /// Allows use of the `:literal` macro fragment specifier (RFC 1576).
    (accepted, macro_literal_matcher, "1.32.0", Some(35625), None),
    /// Allows `macro_rules!` items.
    (accepted, macro_rules, "1.0.0", None, None),
    /// Allows use of the `:vis` macro fragment specifier
    (accepted, macro_vis_matcher, "1.30.0", Some(41022), None),
    /// Allows macro invocations in `extern {}` blocks.
    (accepted, macros_in_extern, "1.40.0", Some(49476), None),
    /// Allows '|' at beginning of match arms (RFC 1925).
    (accepted, match_beginning_vert, "1.25.0", Some(44101), None),
    /// Allows default match binding modes (RFC 2005).
    (accepted, match_default_bindings, "1.26.0", Some(42640), None),
    /// Allows `impl Trait` with multiple unrelated lifetimes.
    (accepted, member_constraints, "1.54.0", Some(61997), None),
    /// Allows the definition of `const fn` functions.
    (accepted, min_const_fn, "1.31.0", Some(53555), None),
    /// The smallest useful subset of const generics.
    (accepted, min_const_generics, "1.51.0", Some(74878), None),
    /// Allows calling `const unsafe fn` inside `unsafe` blocks in `const fn` functions.
    (accepted, min_const_unsafe_fn, "1.33.0", Some(55607), None),
    /// Allows using `Self` and associated types in struct expressions and patterns.
    (accepted, more_struct_aliases, "1.16.0", Some(37544), None),
    /// Allows patterns with concurrent by-move and by-ref bindings.
    /// For example, you can write `Foo(a, ref b)` where `a` is by-move and `b` is by-ref.
    (accepted, move_ref_pattern, "1.49.0", Some(68354), None),
    /// Allows using `#![no_std]`.
    (accepted, no_std, "1.6.0", None, None),
    /// Allows defining identifiers beyond ASCII.
    (accepted, non_ascii_idents, "1.53.0", Some(55467), None),
    /// Allows future-proofing enums/structs with the `#[non_exhaustive]` attribute (RFC 2008).
    (accepted, non_exhaustive, "1.40.0", Some(44109), None),
    /// Allows `foo.rs` as an alternative to `foo/mod.rs`.
    (accepted, non_modrs_mods, "1.30.0", Some(44660), None),
    /// Allows the use of or-patterns (e.g., `0 | 1`).
    (accepted, or_patterns, "1.53.0", Some(54883), None),
    /// Allows annotating functions conforming to `fn(&PanicInfo) -> !` with `#[panic_handler]`.
    /// This defines the behavior of panics.
    (accepted, panic_handler, "1.30.0", Some(44489), None),
    /// Allows attributes in formal function parameters.
    (accepted, param_attrs, "1.39.0", Some(60406), None),
    /// Allows parentheses in patterns.
    (accepted, pattern_parentheses, "1.31.0", Some(51087), None),
    /// Allows procedural macros in `proc-macro` crates.
    (accepted, proc_macro, "1.29.0", Some(38356), None),
    /// Allows multi-segment paths in attributes and derives.
    (accepted, proc_macro_path_invoc, "1.30.0", Some(38356), None),
    /// Allows `pub(restricted)` visibilities (RFC 1422).
    (accepted, pub_restricted, "1.18.0", Some(32409), None),
    /// Allows use of the postfix `?` operator in expressions.
    (accepted, question_mark, "1.13.0", Some(31436), None),
    /// Allows keywords to be escaped for use as identifiers.
    (accepted, raw_identifiers, "1.30.0", Some(48589), None),
    /// Allows relaxing the coherence rules such that
    /// `impl<T> ForeignTrait<LocalType> for ForeignType<T>` is permitted.
    (accepted, re_rebalance_coherence, "1.41.0", Some(55437), None),
    /// Allows numeric fields in struct expressions and patterns.
    (accepted, relaxed_adts, "1.19.0", Some(35626), None),
    /// Lessens the requirements for structs to implement `Unsize`.
    (accepted, relaxed_struct_unsize, "1.58.0", Some(81793), None),
    /// Allows `repr(align(16))` struct attribute (RFC 1358).
    (accepted, repr_align, "1.25.0", Some(33626), None),
    /// Allows using `#[repr(align(X))]` on enums with equivalent semantics
    /// to wrapping an enum in a wrapper struct with `#[repr(align(X))]`.
    (accepted, repr_align_enum, "1.37.0", Some(57996), None),
    /// Allows `#[repr(packed(N))]` attribute on structs.
    (accepted, repr_packed, "1.33.0", Some(33158), None),
    /// Allows `#[repr(transparent)]` attribute on newtype structs.
    (accepted, repr_transparent, "1.28.0", Some(43036), None),
    /// Allows code like `let x: &'static u32 = &42` to work (RFC 1414).
    (accepted, rvalue_static_promotion, "1.21.0", Some(38865), None),
    /// Allows `Self` in type definitions (RFC 2300).
    (accepted, self_in_typedefs, "1.32.0", Some(49303), None),
    /// Allows `Self` struct constructor (RFC 2302).
    (accepted, self_struct_ctor, "1.32.0", Some(51994), None),
    /// Allows using subslice patterns, `[a, .., b]` and `[a, xs @ .., b]`.
    (accepted, slice_patterns, "1.42.0", Some(62254), None),
    /// Allows use of `&foo[a..b]` as a slicing syntax.
    (accepted, slicing_syntax, "1.0.0", None, None),
    /// Allows elision of `'static` lifetimes in `static`s and `const`s.
    (accepted, static_in_const, "1.17.0", Some(35897), None),
    /// Allows the definition recursive static items.
    (accepted, static_recursion, "1.17.0", Some(29719), None),
    /// Allows attributes on struct literal fields.
    (accepted, struct_field_attributes, "1.20.0", Some(38814), None),
    /// Allows struct variants `Foo { baz: u8, .. }` in enums (RFC 418).
    (accepted, struct_variant, "1.0.0", None, None),
    /// Allows `#[target_feature(...)]`.
    (accepted, target_feature, "1.27.0", None, None),
    /// Allows `fn main()` with return types which implements `Termination` (RFC 1937).
    (accepted, termination_trait, "1.26.0", Some(43301), None),
    /// Allows `#[test]` functions where the return type implements `Termination` (RFC 1937).
    (accepted, termination_trait_test, "1.27.0", Some(48854), None),
    /// Allows attributes scoped to tools.
    (accepted, tool_attributes, "1.30.0", Some(44690), None),
    /// Allows scoped lints.
    (accepted, tool_lints, "1.31.0", Some(44690), None),
    /// Allows `#[track_caller]` to be used which provides
    /// accurate caller location reporting during panic (RFC 2091).
    (accepted, track_caller, "1.46.0", Some(47809), None),
    /// Allows #[repr(transparent)] on univariant enums (RFC 2645).
    (accepted, transparent_enums, "1.42.0", Some(60405), None),
    /// Allows indexing tuples.
    (accepted, tuple_indexing, "1.0.0", None, None),
    /// Allows paths to enum variants on type aliases including `Self`.
    (accepted, type_alias_enum_variants, "1.37.0", Some(49683), None),
    /// Allows macros to appear in the type position.
    (accepted, type_macros, "1.13.0", Some(27245), None),
    /// Allows `const _: TYPE = VALUE`.
    (accepted, underscore_const_names, "1.37.0", Some(54912), None),
    /// Allows `use path as _;` and `extern crate c as _;`.
    (accepted, underscore_imports, "1.33.0", Some(48216), None),
    /// Allows `'_` placeholder lifetimes.
    (accepted, underscore_lifetimes, "1.26.0", Some(44524), None),
    /// Allows `use x::y;` to search `x` in the current scope.
    (accepted, uniform_paths, "1.32.0", Some(53130), None),
    /// Allows `impl Trait` in function arguments.
    (accepted, universal_impl_trait, "1.26.0", Some(34511), None),
    /// Allows arbitrary delimited token streams in non-macro attributes.
    (accepted, unrestricted_attribute_tokens, "1.34.0", Some(55208), None),
    /// The `unsafe_op_in_unsafe_fn` lint (allowed by default): no longer treat an unsafe function as an unsafe block.
    (accepted, unsafe_block_in_unsafe_fn, "1.52.0", Some(71668), None),
    /// Allows importing and reexporting macros with `use`,
    /// enables macro modularization in general.
    (accepted, use_extern_macros, "1.30.0", Some(35896), None),
    /// Allows nested groups in `use` items (RFC 2128).
    (accepted, use_nested_groups, "1.25.0", Some(44494), None),
    /// Allows `#[used]` to preserve symbols (see llvm.compiler.used).
    (accepted, used, "1.30.0", Some(40289), None),
    /// Allows the use of `while let` expressions.
    (accepted, while_let, "1.0.0", None, None),
    /// Allows `#![windows_subsystem]`.
    (accepted, windows_subsystem, "1.18.0", Some(37499), None),
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!
    // Features are listed in alphabetical order. Tidy will fail if you don't keep it this way.
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!

    // -------------------------------------------------------------------------
    // feature-group-end: accepted features
    // -------------------------------------------------------------------------
);
