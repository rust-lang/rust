//! List of the accepted feature gates.

use rustc_span::sym;

use super::{Feature, to_nonzero};

macro_rules! declare_features {
    ($(
        $(#[doc = $doc:tt])* (accepted, $feature:ident, $ver:expr, $issue:expr),
    )+) => {
        /// Formerly unstable features that have now been accepted (stabilized).
        pub static ACCEPTED_LANG_FEATURES: &[Feature] = &[
            $(Feature {
                name: sym::$feature,
                since: $ver,
                issue: to_nonzero($issue),
            }),+
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
    (accepted, issue_5723_bootstrap, "1.0.0", None),
    /// These are used to test this portion of the compiler,
    /// they don't actually mean anything.
    (accepted, test_accepted_feature, "1.0.0", None),
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!
    // Features are listed in alphabetical order. Tidy will fail if you don't keep it this way.
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!

    // -------------------------------------------------------------------------
    // feature-group-end: for testing purposes
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // feature-group-start: accepted features
    // -------------------------------------------------------------------------

    // Note that the version indicates when it got *stabilized*.
    // When moving an unstable feature here, set the version number to
    // `CURRENT RUSTC VERSION` with ` ` replaced by `_`.

    /// Allows `#[target_feature(...)]` on aarch64 platforms
    (accepted, aarch64_target_feature, "1.61.0", Some(44839)),
    /// Allows using the `efiapi` ABI.
    (accepted, abi_efiapi, "1.68.0", Some(65815)),
    /// Allows the sysV64 ABI to be specified on all platforms
    /// instead of just the platforms on which it is the C ABI.
    (accepted, abi_sysv64, "1.24.0", Some(36167)),
    /// Allows using the `thiscall` ABI.
    (accepted, abi_thiscall, "1.73.0", None),
    /// Allows using ADX intrinsics from `core::arch::{x86, x86_64}`.
    (accepted, adx_target_feature, "1.61.0", Some(44839)),
    /// Allows explicit discriminants on non-unit enum variants.
    (accepted, arbitrary_enum_discriminant, "1.66.0", Some(60553)),
    /// Allows using `const` operands in inline assembly.
    (accepted, asm_const, "1.82.0", Some(93332)),
    /// Allows using `label` operands in inline assembly.
    (accepted, asm_goto, "1.87.0", Some(119364)),
    /// Allows using `sym` operands in inline assembly.
    (accepted, asm_sym, "1.66.0", Some(93333)),
    /// Allows the definition of associated constants in `trait` or `impl` blocks.
    (accepted, associated_consts, "1.20.0", Some(29646)),
    /// Allows the user of associated type bounds.
    (accepted, associated_type_bounds, "1.79.0", Some(52662)),
    /// Allows using associated `type`s in `trait`s.
    (accepted, associated_types, "1.0.0", None),
    /// Allows free and inherent `async fn`s, `async` blocks, and `<expr>.await` expressions.
    (accepted, async_await, "1.39.0", Some(50547)),
    /// Allows `async || body` closures.
    (accepted, async_closure, "1.85.0", Some(62290)),
    /// Allows async functions to be declared, implemented, and used in traits.
    (accepted, async_fn_in_trait, "1.75.0", Some(91611)),
    /// Allows all literals in attribute lists and values of key-value pairs.
    (accepted, attr_literals, "1.30.0", Some(34981)),
    /// Allows overloading augmented assignment operations like `a += b`.
    (accepted, augmented_assignments, "1.8.0", Some(28235)),
    /// Allows using `avx512*` target features.
    (accepted, avx512_target_feature, "CURRENT_RUSTC_VERSION", Some(44839)),
    /// Allows mixing bind-by-move in patterns and references to those identifiers in guards.
    (accepted, bind_by_move_pattern_guards, "1.39.0", Some(15287)),
    /// Allows bindings in the subpattern of a binding pattern.
    /// For example, you can write `x @ Some(y)`.
    (accepted, bindings_after_at, "1.56.0", Some(65490)),
    /// Allows empty structs and enum variants with braces.
    (accepted, braced_empty_structs, "1.8.0", Some(29720)),
    /// Allows `c"foo"` literals.
    (accepted, c_str_literals, "1.77.0", Some(105723)),
    /// Allows `extern "C-unwind" fn` to enable unwinding across ABI boundaries and treat `extern "C" fn` as nounwind.
    (accepted, c_unwind, "1.81.0", Some(74990)),
    /// Allows `#[cfg_attr(predicate, multiple, attributes, here)]`.
    (accepted, cfg_attr_multi, "1.33.0", Some(54881)),
    /// Allows the use of `#[cfg(<true/false>)]`.
    (accepted, cfg_boolean_literals, "1.88.0", Some(131204)),
    /// Allows the use of `#[cfg(doctest)]`, set when rustdoc is collecting doctests.
    (accepted, cfg_doctest, "1.40.0", Some(62210)),
    /// Enables `#[cfg(panic = "...")]` config key.
    (accepted, cfg_panic, "1.60.0", Some(77443)),
    /// Allows `cfg(target_abi = "...")`.
    (accepted, cfg_target_abi, "1.78.0", Some(80970)),
    /// Allows `cfg(target_feature = "...")`.
    (accepted, cfg_target_feature, "1.27.0", Some(29717)),
    /// Allows `cfg(target_vendor = "...")`.
    (accepted, cfg_target_vendor, "1.33.0", Some(29718)),
    /// Allows implementing `Clone` for closures where possible (RFC 2132).
    (accepted, clone_closures, "1.26.0", Some(44490)),
    /// Allows coercing non capturing closures to function pointers.
    (accepted, closure_to_fn_coercion, "1.19.0", Some(39817)),
    /// Allows using the CMPXCHG16B target feature.
    (accepted, cmpxchg16b_target_feature, "1.69.0", Some(44839)),
    /// Allows use of the `#[collapse_debuginfo]` attribute.
    (accepted, collapse_debuginfo, "1.79.0", Some(100758)),
    /// Allows usage of the `compile_error!` macro.
    (accepted, compile_error, "1.20.0", Some(40872)),
    /// Allows `impl Trait` in function return types.
    (accepted, conservative_impl_trait, "1.26.0", Some(34511)),
    /// Allows calling constructor functions in `const fn`.
    (accepted, const_constructor, "1.40.0", Some(61456)),
    /// Allows the definition of `const extern fn` and `const unsafe extern fn`.
    (accepted, const_extern_fn, "1.83.0", Some(64926)),
    /// Allows basic arithmetic on floating point types in a `const fn`.
    (accepted, const_fn_floating_point_arithmetic, "1.82.0", Some(57241)),
    /// Allows using and casting function pointers in a `const fn`.
    (accepted, const_fn_fn_ptr_basics, "1.61.0", Some(57563)),
    /// Allows trait bounds in `const fn`.
    (accepted, const_fn_trait_bound, "1.61.0", Some(93706)),
    /// Allows calling `transmute` in const fn
    (accepted, const_fn_transmute, "1.56.0", Some(53605)),
    /// Allows accessing fields of unions inside `const` functions.
    (accepted, const_fn_union, "1.56.0", Some(51909)),
    /// Allows unsizing coercions in `const fn`.
    (accepted, const_fn_unsize, "1.54.0", Some(64992)),
    /// Allows const generics to have default values (e.g. `struct Foo<const N: usize = 3>(...);`).
    (accepted, const_generics_defaults, "1.59.0", Some(44580)),
    /// Allows the use of `if` and `match` in constants.
    (accepted, const_if_match, "1.46.0", Some(49146)),
    /// Allows argument and return position `impl Trait` in a `const fn`.
    (accepted, const_impl_trait, "1.61.0", Some(77463)),
    /// Allows indexing into constant arrays.
    (accepted, const_indexing, "1.26.0", Some(29947)),
    /// Allows let bindings, assignments and destructuring in `const` functions and constants.
    /// As long as control flow is not implemented in const eval, `&&` and `||` may not be used
    /// at the same time as let bindings.
    (accepted, const_let, "1.33.0", Some(48821)),
    /// Allows the use of `loop` and `while` in constants.
    (accepted, const_loop, "1.46.0", Some(52000)),
    /// Allows using `&mut` in constant functions.
    (accepted, const_mut_refs, "1.83.0", Some(57349)),
    /// Allows panicking during const eval (producing compile-time errors).
    (accepted, const_panic, "1.57.0", Some(51999)),
    /// Allows dereferencing raw pointers during const eval.
    (accepted, const_raw_ptr_deref, "1.58.0", Some(51911)),
    /// Allows references to types with interior mutability within constants
    (accepted, const_refs_to_cell, "1.83.0", Some(80384)),
    /// Allows creating pointers and references to `static` items in constants.
    (accepted, const_refs_to_static, "1.83.0", Some(119618)),
    /// Allows implementing `Copy` for closures where possible (RFC 2132).
    (accepted, copy_closures, "1.26.0", Some(44490)),
    /// Allows `crate` in paths.
    (accepted, crate_in_paths, "1.30.0", Some(45477)),
    /// Allows users to provide classes for fenced code block using `class:classname`.
    (accepted, custom_code_classes_in_docs, "1.80.0", Some(79483)),
    /// Allows using `#[debugger_visualizer]` attribute.
    (accepted, debugger_visualizer, "1.71.0", Some(95939)),
    /// Allows rustc to inject a default alloc_error_handler
    (accepted, default_alloc_error_handler, "1.68.0", Some(66741)),
    /// Allows using assigning a default type to type parameters in algebraic data type definitions.
    (accepted, default_type_params, "1.0.0", None),
    /// Allows `#[deprecated]` attribute.
    (accepted, deprecated, "1.9.0", Some(29935)),
    /// Allows `#[derive(Default)]` and `#[default]` on enums.
    (accepted, derive_default_enum, "1.62.0", Some(86985)),
    /// Allows the use of destructuring assignments.
    (accepted, destructuring_assignment, "1.59.0", Some(71126)),
    /// Allows using the `#[diagnostic]` attribute tool namespace
    (accepted, diagnostic_namespace, "1.78.0", Some(111996)),
    /// Controls errors in trait implementations.
    (accepted, do_not_recommend, "1.85.0", Some(51992)),
    /// Allows `#[doc(alias = "...")]`.
    (accepted, doc_alias, "1.48.0", Some(50146)),
    /// Allows `..` in tuple (struct) patterns.
    (accepted, dotdot_in_tuple_patterns, "1.14.0", Some(33627)),
    /// Allows `..=` in patterns (RFC 1192).
    (accepted, dotdoteq_in_patterns, "1.26.0", Some(28237)),
    /// Allows `Drop` types in constants (RFC 1440).
    (accepted, drop_types_in_const, "1.22.0", Some(33156)),
    /// Allows using `dyn Trait` as a syntax for trait objects.
    (accepted, dyn_trait, "1.27.0", Some(44662)),
    /// Allows `X..Y` patterns.
    (accepted, exclusive_range_pattern, "1.80.0", Some(37854)),
    /// Allows integer match exhaustiveness checking (RFC 2591).
    (accepted, exhaustive_integer_patterns, "1.33.0", Some(50907)),
    /// Allows explicit generic arguments specification with `impl Trait` present.
    (accepted, explicit_generic_args_with_impl_trait, "1.63.0", Some(83701)),
    /// Uses 2024 rules for matching `expr` fragments in macros. Also enables `expr_2021` fragment.
    (accepted, expr_fragment_specifier_2024, "1.83.0", Some(123742)),
    /// Allows arbitrary expressions in key-value attributes at parse time.
    (accepted, extended_key_value_attributes, "1.54.0", Some(78835)),
    /// Allows resolving absolute paths as paths from other crates.
    (accepted, extern_absolute_paths, "1.30.0", Some(44660)),
    /// Allows `extern crate foo as bar;`. This puts `bar` into extern prelude.
    (accepted, extern_crate_item_prelude, "1.31.0", Some(55599)),
    /// Allows `extern crate self as foo;`.
    /// This puts local crate root into extern prelude under name `foo`.
    (accepted, extern_crate_self, "1.34.0", Some(56409)),
    /// Allows access to crate names passed via `--extern` through prelude.
    (accepted, extern_prelude, "1.30.0", Some(44660)),
    /// Allows using F16C intrinsics from `core::arch::{x86, x86_64}`.
    (accepted, f16c_target_feature, "1.68.0", Some(44839)),
    /// Allows field shorthands (`x` meaning `x: x`) in struct literal expressions.
    (accepted, field_init_shorthand, "1.17.0", Some(37340)),
    /// Allows `#[must_use]` on functions, and introduces must-use operators (RFC 1940).
    (accepted, fn_must_use, "1.27.0", Some(43302)),
    /// Allows capturing variables in scope using format_args!
    (accepted, format_args_capture, "1.58.0", Some(67984)),
    /// Infer generic args for both consts and types.
    (accepted, generic_arg_infer, "CURRENT_RUSTC_VERSION", Some(85077)),
    /// Allows associated types to be generic, e.g., `type Foo<T>;` (RFC 1598).
    (accepted, generic_associated_types, "1.65.0", Some(44265)),
    /// Allows attributes on lifetime/type formal parameters in generics (RFC 1327).
    (accepted, generic_param_attrs, "1.27.0", Some(48848)),
    /// Allows the `#[global_allocator]` attribute.
    (accepted, global_allocator, "1.28.0", Some(27389)),
    // FIXME: explain `globs`.
    (accepted, globs, "1.0.0", None),
    /// Allows using `..=X` as a pattern.
    (accepted, half_open_range_patterns, "1.66.0", Some(67264)),
    /// Allows using the `u128` and `i128` types.
    (accepted, i128_type, "1.26.0", Some(35118)),
    /// Allows the use of `if let` expressions.
    (accepted, if_let, "1.0.0", None),
    /// Rescoping temporaries in `if let` to align with Rust 2024.
    (accepted, if_let_rescope, "1.84.0", Some(124085)),
    /// Allows top level or-patterns (`p | q`) in `if let` and `while let`.
    (accepted, if_while_or_patterns, "1.33.0", Some(48215)),
    /// Allows lifetime elision in `impl` headers. For example:
    /// + `impl<I:Iterator> Iterator for &mut Iterator`
    /// + `impl Debug for Foo<'_>`
    (accepted, impl_header_lifetime_elision, "1.31.0", Some(15872)),
    /// Allows referencing `Self` and projections in impl-trait.
    (accepted, impl_trait_projections, "1.74.0", Some(103532)),
    /// Allows using imported `main` function
    (accepted, imported_main, "1.79.0", Some(28937)),
    /// Allows using `a..=b` and `..=b` as inclusive range syntaxes.
    (accepted, inclusive_range_syntax, "1.26.0", Some(28237)),
    /// Allows inferring outlives requirements (RFC 2093).
    (accepted, infer_outlives_requirements, "1.30.0", Some(44493)),
    /// Allow anonymous constants from an inline `const` block
    (accepted, inline_const, "1.79.0", Some(76001)),
    /// Allows irrefutable patterns in `if let` and `while let` statements (RFC 2086).
    (accepted, irrefutable_let_patterns, "1.33.0", Some(44495)),
    /// Allows `#[instruction_set(_)]` attribute.
    (accepted, isa_attribute, "1.67.0", Some(74727)),
    /// Allows some increased flexibility in the name resolution rules,
    /// especially around globs and shadowing (RFC 1560).
    (accepted, item_like_imports, "1.15.0", Some(35120)),
    // Allows using the `kl` and `widekl` target features and the associated intrinsics
    (accepted, keylocker_x86, "CURRENT_RUSTC_VERSION", Some(134813)),
    /// Allows `'a: { break 'a; }`.
    (accepted, label_break_value, "1.65.0", Some(48594)),
    /// Allows `let...else` statements.
    (accepted, let_else, "1.65.0", Some(87335)),
    /// Allows using `reason` in lint attributes and the `#[expect(lint)]` lint check.
    (accepted, lint_reasons, "1.81.0", Some(54503)),
    /// Allows `break {expr}` with a value inside `loop`s.
    (accepted, loop_break_value, "1.19.0", Some(37339)),
    /// Allows use of `?` as the Kleene "at most one" operator in macros.
    (accepted, macro_at_most_once_rep, "1.32.0", Some(48075)),
    /// Allows macro attributes to observe output of `#[derive]`.
    (accepted, macro_attributes_in_derive_output, "1.57.0", Some(81119)),
    /// Allows use of the `:lifetime` macro fragment specifier.
    (accepted, macro_lifetime_matcher, "1.27.0", Some(34303)),
    /// Allows use of the `:literal` macro fragment specifier (RFC 1576).
    (accepted, macro_literal_matcher, "1.32.0", Some(35625)),
    /// Allows `macro_rules!` items.
    (accepted, macro_rules, "1.0.0", None),
    /// Allows use of the `:vis` macro fragment specifier
    (accepted, macro_vis_matcher, "1.30.0", Some(41022)),
    /// Allows macro invocations in `extern {}` blocks.
    (accepted, macros_in_extern, "1.40.0", Some(49476)),
    /// Allows '|' at beginning of match arms (RFC 1925).
    (accepted, match_beginning_vert, "1.25.0", Some(44101)),
    /// Allows default match binding modes (RFC 2005).
    (accepted, match_default_bindings, "1.26.0", Some(42640)),
    /// Allows `impl Trait` with multiple unrelated lifetimes.
    (accepted, member_constraints, "1.54.0", Some(61997)),
    /// Allows the definition of `const fn` functions.
    (accepted, min_const_fn, "1.31.0", Some(53555)),
    /// The smallest useful subset of const generics.
    (accepted, min_const_generics, "1.51.0", Some(74878)),
    /// Allows calling `const unsafe fn` inside `unsafe` blocks in `const fn` functions.
    (accepted, min_const_unsafe_fn, "1.33.0", Some(55607)),
    /// Allows exhaustive pattern matching on uninhabited types when matched by value.
    (accepted, min_exhaustive_patterns, "1.82.0", Some(119612)),
    /// Allows using `Self` and associated types in struct expressions and patterns.
    (accepted, more_struct_aliases, "1.16.0", Some(37544)),
    /// Allows using the MOVBE target feature.
    (accepted, movbe_target_feature, "1.70.0", Some(44839)),
    /// Allows patterns with concurrent by-move and by-ref bindings.
    /// For example, you can write `Foo(a, ref b)` where `a` is by-move and `b` is by-ref.
    (accepted, move_ref_pattern, "1.49.0", Some(68354)),
    /// Allows using `#[naked]` on functions.
    (accepted, naked_functions, "1.88.0", Some(90957)),
    /// Allows specifying modifiers in the link attribute: `#[link(modifiers = "...")]`
    (accepted, native_link_modifiers, "1.61.0", Some(81490)),
    /// Allows specifying the bundle link modifier
    (accepted, native_link_modifiers_bundle, "1.63.0", Some(81490)),
    /// Allows specifying the verbatim link modifier
    (accepted, native_link_modifiers_verbatim, "1.67.0", Some(81490)),
    /// Allows specifying the whole-archive link modifier
    (accepted, native_link_modifiers_whole_archive, "1.61.0", Some(81490)),
    /// Allows using non lexical lifetimes (RFC 2094).
    (accepted, nll, "1.63.0", Some(43234)),
    /// Allows using `#![no_std]`.
    (accepted, no_std, "1.6.0", None),
    /// Allows defining identifiers beyond ASCII.
    (accepted, non_ascii_idents, "1.53.0", Some(55467)),
    /// Allows future-proofing enums/structs with the `#[non_exhaustive]` attribute (RFC 2008).
    (accepted, non_exhaustive, "1.40.0", Some(44109)),
    /// Allows `foo.rs` as an alternative to `foo/mod.rs`.
    (accepted, non_modrs_mods, "1.30.0", Some(44660)),
    /// Allows using multiple nested field accesses in offset_of!
    (accepted, offset_of_nested, "1.82.0", Some(120140)),
    /// Allows the use of or-patterns (e.g., `0 | 1`).
    (accepted, or_patterns, "1.53.0", Some(54883)),
    /// Allows using `+bundle,+whole-archive` link modifiers with native libs.
    (accepted, packed_bundled_libs, "1.74.0", Some(108081)),
    /// Allows annotating functions conforming to `fn(&PanicInfo) -> !` with `#[panic_handler]`.
    /// This defines the behavior of panics.
    (accepted, panic_handler, "1.30.0", Some(44489)),
    /// Allows attributes in formal function parameters.
    (accepted, param_attrs, "1.39.0", Some(60406)),
    /// Allows parentheses in patterns.
    (accepted, pattern_parentheses, "1.31.0", Some(51087)),
    /// Allows `use<'a, 'b, A, B>` in `impl Trait + use<...>` for precise capture of generic args.
    (accepted, precise_capturing, "1.82.0", Some(123432)),
    /// Allows `use<..>` precise capturign on impl Trait in traits.
    (accepted, precise_capturing_in_traits, "1.87.0", Some(130044)),
    /// Allows procedural macros in `proc-macro` crates.
    (accepted, proc_macro, "1.29.0", Some(38356)),
    /// Allows multi-segment paths in attributes and derives.
    (accepted, proc_macro_path_invoc, "1.30.0", Some(38356)),
    /// Allows `pub(restricted)` visibilities (RFC 1422).
    (accepted, pub_restricted, "1.18.0", Some(32409)),
    /// Allows use of the postfix `?` operator in expressions.
    (accepted, question_mark, "1.13.0", Some(31436)),
    /// Allows the use of raw-dylibs (RFC 2627).
    (accepted, raw_dylib, "1.71.0", Some(58713)),
    /// Allows keywords to be escaped for use as identifiers.
    (accepted, raw_identifiers, "1.30.0", Some(48589)),
    /// Allows `&raw const $place_expr` and `&raw mut $place_expr` expressions.
    (accepted, raw_ref_op, "1.82.0", Some(64490)),
    /// Allows relaxing the coherence rules such that
    /// `impl<T> ForeignTrait<LocalType> for ForeignType<T>` is permitted.
    (accepted, re_rebalance_coherence, "1.41.0", Some(55437)),
    /// Allows numeric fields in struct expressions and patterns.
    (accepted, relaxed_adts, "1.19.0", Some(35626)),
    /// Lessens the requirements for structs to implement `Unsize`.
    (accepted, relaxed_struct_unsize, "1.58.0", Some(81793)),
    /// Allows the `#[repr(i128)]` attribute for enums.
    (accepted, repr128, "CURRENT_RUSTC_VERSION", Some(56071)),
    /// Allows `repr(align(16))` struct attribute (RFC 1358).
    (accepted, repr_align, "1.25.0", Some(33626)),
    /// Allows using `#[repr(align(X))]` on enums with equivalent semantics
    /// to wrapping an enum in a wrapper struct with `#[repr(align(X))]`.
    (accepted, repr_align_enum, "1.37.0", Some(57996)),
    /// Allows `#[repr(packed(N))]` attribute on structs.
    (accepted, repr_packed, "1.33.0", Some(33158)),
    /// Allows `#[repr(transparent)]` attribute on newtype structs.
    (accepted, repr_transparent, "1.28.0", Some(43036)),
    /// Allows enums like Result<T, E> to be used across FFI, if T's niche value can
    /// be used to describe E or vice-versa.
    (accepted, result_ffi_guarantees, "1.84.0", Some(110503)),
    /// Allows return-position `impl Trait` in traits.
    (accepted, return_position_impl_trait_in_trait, "1.75.0", Some(91611)),
    /// Allows code like `let x: &'static u32 = &42` to work (RFC 1414).
    (accepted, rvalue_static_promotion, "1.21.0", Some(38865)),
    /// Allows `Self` in type definitions (RFC 2300).
    (accepted, self_in_typedefs, "1.32.0", Some(49303)),
    /// Allows `Self` struct constructor (RFC 2302).
    (accepted, self_struct_ctor, "1.32.0", Some(51994)),
    /// Allows use of x86 SHA512, SM3 and SM4 target-features and intrinsics
    (accepted, sha512_sm_x86, "CURRENT_RUSTC_VERSION", Some(126624)),
    /// Shortern the tail expression lifetime
    (accepted, shorter_tail_lifetimes, "1.84.0", Some(123739)),
    /// Allows using subslice patterns, `[a, .., b]` and `[a, xs @ .., b]`.
    (accepted, slice_patterns, "1.42.0", Some(62254)),
    /// Allows use of `&foo[a..b]` as a slicing syntax.
    (accepted, slicing_syntax, "1.0.0", None),
    /// Allows elision of `'static` lifetimes in `static`s and `const`s.
    (accepted, static_in_const, "1.17.0", Some(35897)),
    /// Allows the definition recursive static items.
    (accepted, static_recursion, "1.17.0", Some(29719)),
    /// Allows attributes on struct literal fields.
    (accepted, struct_field_attributes, "1.20.0", Some(38814)),
    /// Allows struct variants `Foo { baz: u8, .. }` in enums (RFC 418).
    (accepted, struct_variant, "1.0.0", None),
    /// Allows `#[target_feature(...)]`.
    (accepted, target_feature, "1.27.0", None),
    /// Allows the use of `#[target_feature]` on safe functions.
    (accepted, target_feature_11, "1.86.0", Some(69098)),
    /// Allows `fn main()` with return types which implements `Termination` (RFC 1937).
    (accepted, termination_trait, "1.26.0", Some(43301)),
    /// Allows `#[test]` functions where the return type implements `Termination` (RFC 1937).
    (accepted, termination_trait_test, "1.27.0", Some(48854)),
    /// Allows attributes scoped to tools.
    (accepted, tool_attributes, "1.30.0", Some(44690)),
    /// Allows scoped lints.
    (accepted, tool_lints, "1.31.0", Some(44690)),
    /// Allows `#[track_caller]` to be used which provides
    /// accurate caller location reporting during panic (RFC 2091).
    (accepted, track_caller, "1.46.0", Some(47809)),
    /// Allows dyn upcasting trait objects via supertraits.
    /// Dyn upcasting is casting, e.g., `dyn Foo -> dyn Bar` where `Foo: Bar`.
    (accepted, trait_upcasting, "1.86.0", Some(65991)),
    /// Allows #[repr(transparent)] on univariant enums (RFC 2645).
    (accepted, transparent_enums, "1.42.0", Some(60405)),
    /// Allows indexing tuples.
    (accepted, tuple_indexing, "1.0.0", None),
    /// Allows paths to enum variants on type aliases including `Self`.
    (accepted, type_alias_enum_variants, "1.37.0", Some(49683)),
    /// Allows macros to appear in the type position.
    (accepted, type_macros, "1.13.0", Some(27245)),
    /// Allows using type privacy lints (`private_interfaces`, `private_bounds`, `unnameable_types`).
    (accepted, type_privacy_lints, "1.79.0", Some(48054)),
    /// Allows `const _: TYPE = VALUE`.
    (accepted, underscore_const_names, "1.37.0", Some(54912)),
    /// Allows `use path as _;` and `extern crate c as _;`.
    (accepted, underscore_imports, "1.33.0", Some(48216)),
    /// Allows `'_` placeholder lifetimes.
    (accepted, underscore_lifetimes, "1.26.0", Some(44524)),
    /// Allows `use x::y;` to search `x` in the current scope.
    (accepted, uniform_paths, "1.32.0", Some(53130)),
    /// Allows `impl Trait` in function arguments.
    (accepted, universal_impl_trait, "1.26.0", Some(34511)),
    /// Allows arbitrary delimited token streams in non-macro attributes.
    (accepted, unrestricted_attribute_tokens, "1.34.0", Some(55208)),
    /// Allows unsafe attributes.
    (accepted, unsafe_attributes, "1.82.0", Some(123757)),
    /// The `unsafe_op_in_unsafe_fn` lint (allowed by default): no longer treat an unsafe function as an unsafe block.
    (accepted, unsafe_block_in_unsafe_fn, "1.52.0", Some(71668)),
    /// Allows unsafe on extern declarations and safety qualifiers over internal items.
    (accepted, unsafe_extern_blocks, "1.82.0", Some(123743)),
    /// Allows importing and reexporting macros with `use`,
    /// enables macro modularization in general.
    (accepted, use_extern_macros, "1.30.0", Some(35896)),
    /// Allows nested groups in `use` items (RFC 2128).
    (accepted, use_nested_groups, "1.25.0", Some(44494)),
    /// Allows `#[used]` to preserve symbols (see llvm.compiler.used).
    (accepted, used, "1.30.0", Some(40289)),
    /// Allows the use of `while let` expressions.
    (accepted, while_let, "1.0.0", None),
    /// Allows `#![windows_subsystem]`.
    (accepted, windows_subsystem, "1.18.0", Some(37499)),
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!
    // Features are listed in alphabetical order. Tidy will fail if you don't keep it this way.
    // !!!!    !!!!    !!!!    !!!!   !!!!    !!!!    !!!!    !!!!    !!!!    !!!!    !!!!

    // -------------------------------------------------------------------------
    // feature-group-end: accepted features
    // -------------------------------------------------------------------------
);
