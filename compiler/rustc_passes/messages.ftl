-passes_previously_accepted =
    this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

-passes_see_issue =
    see issue #{$issue} <https://github.com/rust-lang/rust/issues/{$issue}> for more information

passes_abi_invalid_attribute =
    `#[rustc_abi]` can only be applied to function items, type aliases, and associated functions
passes_abi_ne =
    ABIs are not compatible
    left ABI = {$left}
    right ABI = {$right}
passes_abi_of =
    fn_abi_of({$fn_name}) = {$fn_abi}

passes_align_should_be_repr_align =
    `#[align(...)]` is not supported on {$item} items
    .suggestion = use `#[repr(align(...))]` instead

passes_allow_incoherent_impl =
    `rustc_allow_incoherent_impl` attribute should be applied to impl items
    .label = the only currently supported targets are inherent methods

passes_allow_internal_unstable =
    attribute should be applied to a macro
    .label = not a macro

passes_attr_application_enum =
    attribute should be applied to an enum
    .label = not an enum

passes_attr_application_struct =
    attribute should be applied to a struct
    .label = not a struct

passes_attr_application_struct_enum_union =
    attribute should be applied to a struct, enum, or union
    .label = not a struct, enum, or union

passes_attr_application_struct_union =
    attribute should be applied to a struct or union
    .label = not a struct or union

passes_attr_crate_level =
    this attribute can only be applied at the crate level
    .suggestion = to apply to the crate, use an inner attribute
    .note = read <https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#at-the-crate-level> for more information

passes_attr_only_in_functions =
    `{$attr}` attribute can only be used on functions

passes_autodiff_attr =
    `#[autodiff]` should be applied to a function
    .label = not a function

passes_both_ffi_const_and_pure =
    `#[ffi_const]` function cannot be `#[ffi_pure]`

passes_cannot_stabilize_deprecated =
    an API can't be stabilized after it is deprecated
    .label = invalid version
    .item = the stability attribute annotates this item

passes_change_fields_to_be_of_unit_type =
    consider changing the { $num ->
      [one] field
     *[other] fields
    } to be of unit type to suppress this warning while preserving the field numbering, or remove the { $num ->
      [one] field
     *[other] fields
    }

passes_cold =
    {passes_should_be_applied_to_fn}
    .warn = {-passes_previously_accepted}
    .label = {passes_should_be_applied_to_fn.label}

passes_collapse_debuginfo =
    `collapse_debuginfo` attribute should be applied to macro definitions
    .label = not a macro definition

passes_confusables = attribute should be applied to an inherent method
    .label = not an inherent method

passes_const_continue_attr =
    `#[const_continue]` should be applied to a break expression
    .label = not a break expression

passes_const_stable_not_stable =
    attribute `#[rustc_const_stable]` can only be applied to functions that are declared `#[stable]`
    .label = attribute specified here

passes_coroutine_on_non_closure =
    attribute should be applied to closures
    .label = not a closure

passes_coverage_attribute_not_allowed =
    coverage attribute not allowed here
    .not_fn_impl_mod = not a function, impl block, or module
    .no_body = function has no body
    .help = coverage attribute can be applied to a function (with body), impl block, or module

passes_dead_codes =
    { $multiple ->
      *[true] multiple {$descr}s are
       [false] { $num ->
         [one] {$descr} {$name_list} is
        *[other] {$descr}s {$name_list} are
       }
    } never {$participle}

passes_debug_visualizer_invalid =
    invalid argument
    .note_1 = expected: `natvis_file = "..."`
    .note_2 = OR
    .note_3 = expected: `gdb_script_file = "..."`

passes_debug_visualizer_placement =
    attribute should be applied to a module

passes_debug_visualizer_unreadable =
    couldn't read {$file}: {$error}

passes_deprecated =
    attribute is ignored here

passes_deprecated_annotation_has_no_effect =
    this `#[deprecated]` annotation has no effect
    .suggestion = remove the unnecessary deprecation attribute

passes_deprecated_attribute =
    deprecated attribute must be paired with either stable or unstable attribute

passes_diagnostic_diagnostic_on_unimplemented_only_for_traits =
    `#[diagnostic::on_unimplemented]` can only be applied to trait definitions

passes_diagnostic_item_first_defined =
    the diagnostic item is first defined here

passes_doc_alias_bad_char =
    {$char_} character isn't allowed in {$attr_str}

passes_doc_alias_bad_location =
    {$attr_str} isn't allowed on {$location}

passes_doc_alias_duplicated = doc alias is duplicated
    .label = first defined here

passes_doc_alias_empty =
    {$attr_str} attribute cannot have empty value

passes_doc_alias_malformed =
    doc alias attribute expects a string `#[doc(alias = "a")]` or a list of strings `#[doc(alias("a", "b"))]`

passes_doc_alias_not_an_alias =
    {$attr_str} is the same as the item's name

passes_doc_alias_not_string_literal =
    `#[doc(alias("a"))]` expects string literals

passes_doc_alias_start_end =
    {$attr_str} cannot start or end with ' '

passes_doc_attr_not_crate_level =
    `#![doc({$attr_name} = "...")]` isn't allowed as a crate-level attribute

passes_doc_cfg_hide_takes_list =
    `#[doc(cfg_hide(...))]` takes a list of attributes

passes_doc_expect_str =
    doc {$attr_name} attribute expects a string: #[doc({$attr_name} = "a")]

passes_doc_fake_variadic_not_valid =
    `#[doc(fake_variadic)]` must be used on the first of a set of tuple or fn pointer trait impls with varying arity

passes_doc_inline_conflict =
    conflicting doc inlining attributes
    .help = remove one of the conflicting attributes

passes_doc_inline_conflict_first =
    this attribute...

passes_doc_inline_conflict_second =
    {"."}..conflicts with this attribute

passes_doc_inline_only_use =
    this attribute can only be applied to a `use` item
    .label = only applicable on `use` items
    .not_a_use_item_label = not a `use` item
    .note = read <https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#inline-and-no_inline> for more information

passes_doc_invalid =
    invalid `doc` attribute

passes_doc_keyword_empty_mod =
    `#[doc(keyword = "...")]` should be used on empty modules

passes_doc_keyword_not_keyword =
    nonexistent keyword `{$keyword}` used in `#[doc(keyword = "...")]`
    .help = only existing keywords are allowed in core/std

passes_doc_keyword_not_mod =
    `#[doc(keyword = "...")]` should be used on modules

passes_doc_keyword_only_impl =
    `#[doc(keyword = "...")]` should be used on impl blocks

passes_doc_masked_not_extern_crate_self =
    this attribute cannot be applied to an `extern crate self` item
    .label = not applicable on `extern crate self` items
    .extern_crate_self_label = `extern crate self` defined here

passes_doc_masked_only_extern_crate =
    this attribute can only be applied to an `extern crate` item
    .label = only applicable on `extern crate` items
    .not_an_extern_crate_label = not an `extern crate` item
    .note = read <https://doc.rust-lang.org/unstable-book/language-features/doc-masked.html> for more information

passes_doc_rust_logo =
    the `#[doc(rust_logo)]` attribute is used for Rust branding

passes_doc_search_unbox_invalid =
    `#[doc(search_unbox)]` should be used on generic structs and enums

passes_doc_test_literal = `#![doc(test(...)]` does not take a literal

passes_doc_test_takes_list =
    `#[doc(test(...)]` takes a list of attributes

passes_doc_test_unknown =
    unknown `doc(test)` attribute `{$path}`

passes_doc_test_unknown_any =
    unknown `doc` attribute `{$path}`

passes_doc_test_unknown_include =
    unknown `doc` attribute `{$path}`
    .suggestion = use `doc = include_str!` instead

passes_doc_test_unknown_passes =
    unknown `doc` attribute `{$path}`
    .note = `doc` attribute `{$path}` no longer functions; see issue #44136 <https://github.com/rust-lang/rust/issues/44136>
    .label = no longer functions
    .help = you may want to use `doc(document_private_items)`
    .no_op_note = `doc({$path})` is now a no-op

passes_doc_test_unknown_plugins =
    unknown `doc` attribute `{$path}`
    .note = `doc` attribute `{$path}` no longer functions; see issue #44136 <https://github.com/rust-lang/rust/issues/44136> and CVE-2018-1000622 <https://nvd.nist.gov/vuln/detail/CVE-2018-1000622>
    .label = no longer functions
    .no_op_note = `doc({$path})` is now a no-op

passes_doc_test_unknown_spotlight =
    unknown `doc` attribute `{$path}`
    .note = `doc(spotlight)` was renamed to `doc(notable_trait)`
    .suggestion = use `notable_trait` instead
    .no_op_note = `doc(spotlight)` is now a no-op

passes_duplicate_diagnostic_item_in_crate =
    duplicate diagnostic item in crate `{$crate_name}`: `{$name}`
    .note = the diagnostic item is first defined in crate `{$orig_crate_name}`

passes_duplicate_feature_err =
    the feature `{$feature}` has already been enabled

passes_duplicate_lang_item =
    found duplicate lang item `{$lang_item_name}`
    .first_defined_span = the lang item is first defined here
    .first_defined_crate_depends = the lang item is first defined in crate `{$orig_crate_name}` (which `{$orig_dependency_of}` depends on)
    .first_defined_crate = the lang item is first defined in crate `{$orig_crate_name}`
    .first_definition_local = first definition in the local crate (`{$orig_crate_name}`)
    .second_definition_local = second definition in the local crate (`{$crate_name}`)
    .first_definition_path = first definition in `{$orig_crate_name}` loaded from {$orig_path}
    .second_definition_path = second definition in `{$crate_name}` loaded from {$path}

passes_duplicate_lang_item_crate =
    duplicate lang item in crate `{$crate_name}`: `{$lang_item_name}`
    .first_defined_span = the lang item is first defined here
    .first_defined_crate_depends = the lang item is first defined in crate `{$orig_crate_name}` (which `{$orig_dependency_of}` depends on)
    .first_defined_crate = the lang item is first defined in crate `{$orig_crate_name}`
    .first_definition_local = first definition in the local crate (`{$orig_crate_name}`)
    .second_definition_local = second definition in the local crate (`{$crate_name}`)
    .first_definition_path = first definition in `{$orig_crate_name}` loaded from {$orig_path}
    .second_definition_path = second definition in `{$crate_name}` loaded from {$path}

passes_duplicate_lang_item_crate_depends =
    duplicate lang item in crate `{$crate_name}` (which `{$dependency_of}` depends on): `{$lang_item_name}`
    .first_defined_span = the lang item is first defined here
    .first_defined_crate_depends = the lang item is first defined in crate `{$orig_crate_name}` (which `{$orig_dependency_of}` depends on)
    .first_defined_crate = the lang item is first defined in crate `{$orig_crate_name}`
    .first_definition_local = first definition in the local crate (`{$orig_crate_name}`)
    .second_definition_local = second definition in the local crate (`{$crate_name}`)
    .first_definition_path = first definition in `{$orig_crate_name}` loaded from {$orig_path}
    .second_definition_path = second definition in `{$crate_name}` loaded from {$path}

passes_enum_variant_same_name =
    it is impossible to refer to the {$dead_descr} `{$dead_name}` because it is shadowed by this enum variant with the same name

passes_export_name =
    attribute should be applied to a free function, impl method or static
    .label = not a free function, impl method or static

passes_extern_main =
    the `main` function cannot be declared in an `extern` block

passes_feature_previously_declared =
    feature `{$feature}` is declared {$declared}, but was previously declared {$prev_declared}

passes_feature_stable_twice =
    feature `{$feature}` is declared stable since {$since}, but was previously declared stable since {$prev_since}

passes_ffi_const_invalid_target =
    `#[ffi_const]` may only be used on foreign functions

passes_ffi_pure_invalid_target =
    `#[ffi_pure]` may only be used on foreign functions

passes_has_incoherent_inherent_impl =
    `rustc_has_incoherent_inherent_impls` attribute should be applied to types or traits
    .label = only adts, extern types and traits are supported

passes_ignored_attr =
    `#[{$sym}]` is ignored on struct fields and match arms
    .warn = {-passes_previously_accepted}
    .note = {-passes_see_issue(issue: "80564")}

passes_ignored_attr_with_macro =
    `#[{$sym}]` is ignored on struct fields, match arms and macro defs
    .warn = {-passes_previously_accepted}
    .note = {-passes_see_issue(issue: "80564")}

passes_ignored_derived_impls =
    `{$name}` has {$trait_list_len ->
      [one] a derived impl
     *[other] derived impls
    } for the {$trait_list_len ->
      [one] trait {$trait_list}, but this is
     *[other] traits {$trait_list}, but these are
    } intentionally ignored during dead code analysis

passes_implied_feature_not_exist =
    feature `{$implied_by}` implying `{$feature}` does not exist

passes_incorrect_crate_type = lang items are not allowed in stable dylibs

passes_incorrect_do_not_recommend_args =
    `#[diagnostic::do_not_recommend]` does not expect any arguments

passes_incorrect_do_not_recommend_location =
    `#[diagnostic::do_not_recommend]` can only be placed on trait implementations

passes_incorrect_target =
    `{$name}` lang item must be applied to a {$kind} with {$at_least ->
        [true] at least {$num}
        *[false] {$num}
    } generic {$num ->
        [one] argument
        *[other] arguments
    }
    .label = this {$kind} has {$actual_num} generic {$actual_num ->
        [one] argument
        *[other] arguments
    }

passes_ineffective_unstable_impl = an `#[unstable]` annotation here has no effect
    .note = see issue #55436 <https://github.com/rust-lang/rust/issues/55436> for more information

passes_inline_ignored_constants =
    `#[inline]` is ignored on constants
    .warn = {-passes_previously_accepted}
    .note = {-passes_see_issue(issue: "65833")}

passes_inline_ignored_for_exported =
    `#[inline]` is ignored on externally exported functions
    .help = externally exported functions are functions with `#[no_mangle]`, `#[export_name]`, or `#[linkage]`

passes_inline_ignored_function_prototype =
    `#[inline]` is ignored on function prototypes

passes_inline_not_fn_or_closure =
    attribute should be applied to function or closure
    .label = not a function or closure

passes_inner_crate_level_attr =
    crate-level attribute should be in the root module

passes_invalid_attr_at_crate_level =
    `{$name}` attribute cannot be used at crate level
    .suggestion = perhaps you meant to use an outer attribute

passes_invalid_attr_at_crate_level_item =
    the inner attribute doesn't annotate this {$kind}

passes_invalid_macro_export_arguments = invalid `#[macro_export]` argument

passes_invalid_macro_export_arguments_too_many_items = `#[macro_export]` can only take 1 or 0 arguments

passes_lang_item_fn = {$name ->
    [panic_impl] `#[panic_handler]`
    *[other] `{$name}` lang item
} function

passes_lang_item_fn_with_target_feature =
    {passes_lang_item_fn} is not allowed to have `#[target_feature]`
    .label = {passes_lang_item_fn} is not allowed to have `#[target_feature]`

passes_lang_item_fn_with_track_caller =
    {passes_lang_item_fn} is not allowed to have `#[track_caller]`
    .label = {passes_lang_item_fn} is not allowed to have `#[track_caller]`

passes_lang_item_on_incorrect_target =
    `{$name}` lang item must be applied to a {$expected_target}
    .label = attribute should be applied to a {$expected_target}, not a {$actual_target}

passes_layout_abi =
    abi: {$abi}
passes_layout_align =
    align: {$align}
passes_layout_homogeneous_aggregate =
    homogeneous_aggregate: {$homogeneous_aggregate}
passes_layout_invalid_attribute =
    `#[rustc_layout]` can only be applied to `struct`/`enum`/`union` declarations and type aliases
passes_layout_of =
    layout_of({$normalized_ty}) = {$ty_layout}
passes_layout_size =
    size: {$size}

passes_link =
    attribute should be applied to an `extern` block with non-Rust ABI
    .warn = {-passes_previously_accepted}
    .label = not an `extern` block

passes_link_name =
    attribute should be applied to a foreign function or static
    .warn = {-passes_previously_accepted}
    .label = not a foreign function or static
    .help = try `#[link(name = "{$value}")]` instead

passes_link_ordinal =
    attribute should be applied to a foreign function or static
    .label = not a foreign function or static

passes_link_section =
    attribute should be applied to a function or static
    .warn = {-passes_previously_accepted}
    .label = not a function or static

passes_linkage =
    attribute should be applied to a function or static
    .label = not a function definition or static

passes_loop_match_attr =
    `#[loop_match]` should be applied to a loop
    .label = not a loop

passes_macro_export =
    `#[macro_export]` only has an effect on macro definitions

passes_macro_export_on_decl_macro =
    `#[macro_export]` has no effect on declarative macro definitions
    .note = declarative macros follow the same exporting rules as regular items

passes_macro_use =
    `#[{$name}]` only has an effect on `extern crate` and modules

passes_may_dangle =
    `#[may_dangle]` must be applied to a lifetime or type generic parameter in `Drop` impl

passes_maybe_string_interpolation = you might have meant to use string interpolation in this string literal

passes_missing_const_err =
    attributes `#[rustc_const_unstable]`, `#[rustc_const_stable]` and `#[rustc_const_stable_indirect]` require the function or method to be `const`
    .help = make the function or method const

passes_missing_const_stab_attr =
    {$descr} has missing const stability attribute

passes_missing_lang_item =
    lang item required, but not found: `{$name}`
    .note = this can occur when a binary crate with `#![no_std]` is compiled for a target where `{$name}` is defined in the standard library
    .help = you may be able to compile for a target that doesn't need `{$name}`, specify a target with `--target` or in `.cargo/config`

passes_missing_panic_handler =
    `#[panic_handler]` function required, but not found

passes_missing_stability_attr =
    {$descr} has missing stability attribute

passes_multiple_rustc_main =
    multiple functions with a `#[rustc_main]` attribute
    .first = first `#[rustc_main]` function
    .additional = additional `#[rustc_main]` function

passes_must_not_suspend =
    `must_not_suspend` attribute should be applied to a struct, enum, union, or trait
    .label = is not a struct, enum, union, or trait

passes_must_use_no_effect =
    `#[must_use]` has no effect when applied to {$article} {$target}

passes_no_link =
    attribute should be applied to an `extern crate` item
    .label = not an `extern crate` item

passes_no_main_function =
    `main` function not found in crate `{$crate_name}`
    .here_is_main = here is a function named `main`
    .one_or_more_possible_main = you have one or more functions named `main` not defined at the crate level
    .consider_moving_main = consider moving the `main` function definitions
    .main_must_be_defined_at_crate = the main function must be defined at the crate level{$has_filename ->
        [true] {" "}(in `{$filename}`)
        *[false] {""}
    }
    .consider_adding_main_to_file = consider adding a `main` function to `{$filename}`
    .consider_adding_main_at_crate = consider adding a `main` function at the crate level
    .teach_note = If you don't know the basics of Rust, you can go look to the Rust Book to get started: https://doc.rust-lang.org/book/
    .non_function_main = non-function item at `crate::main` is found

passes_no_mangle =
    attribute should be applied to a free function, impl method or static
    .warn = {-passes_previously_accepted}
    .label = not a free function, impl method or static

passes_no_mangle_foreign =
    `#[no_mangle]` has no effect on a foreign {$foreign_item_kind}
    .warn = {-passes_previously_accepted}
    .label = foreign {$foreign_item_kind}
    .note = symbol names in extern blocks are not mangled
    .suggestion = remove this attribute

passes_no_sanitize =
    `#[no_sanitize({$attr_str})]` should be applied to {$accepted_kind}
    .label = not {$accepted_kind}

passes_non_exaustive_with_default_field_values =
    `#[non_exhaustive]` can't be used to annotate items with default field values
    .label = this struct has default field values

passes_non_exported_macro_invalid_attrs =
    attribute should be applied to function or closure
    .label = not a function or closure

passes_object_lifetime_err =
    {$repr}

passes_only_has_effect_on =
    `#[{$attr_name}]` only has an effect on {$target_name ->
        [function] functions
        [module] modules
        [implementation_block] implementation blocks
        *[unspecified] (unspecified--this is a compiler bug)
    }

passes_optimize_invalid_target =
    attribute applied to an invalid target
    .label = invalid target

passes_outer_crate_level_attr =
    crate-level attribute should be an inner attribute: add an exclamation mark: `#![foo]`


passes_panic_unwind_without_std =
    unwinding panics are not supported without std
    .note = since the core library is usually precompiled with panic="unwind", rebuilding your crate with panic="abort" may not be enough to fix the problem
    .help = using nightly cargo, use -Zbuild-std with panic="abort" to avoid unwinding

passes_parent_info =
    {$num ->
      [one] {$descr}
     *[other] {$descr}s
    } in this {$parent_descr}

passes_pass_by_value =
    `pass_by_value` attribute should be applied to a struct, enum or type alias
    .label = is not a struct, enum or type alias

passes_proc_macro_bad_sig = {$kind} has incorrect signature

passes_remove_fields =
    consider removing { $num ->
      [one] this
     *[other] these
    } { $num ->
      [one] field
     *[other] fields
    }

passes_repr_align_greater_than_target_max =
    alignment must not be greater than `isize::MAX` bytes
    .note = `isize::MAX` is {$size} for the current target

passes_repr_align_should_be_align =
    `#[repr(align(...))]` is not supported on {$item} items
    .help = use `#[align(...)]` instead

passes_repr_conflicting =
    conflicting representation hints

passes_rustc_allow_const_fn_unstable =
    attribute should be applied to `const fn`
    .label = not a `const fn`

passes_rustc_const_stable_indirect_pairing =
    `const_stable_indirect` attribute does not make sense on `rustc_const_stable` function, its behavior is already implied
passes_rustc_dirty_clean =
    attribute requires -Z query-dep-graph to be enabled

passes_rustc_force_inline =
    attribute should be applied to a function
    .label = not a function definition

passes_rustc_force_inline_coro =
    attribute cannot be applied to a `async`, `gen` or `async gen` function
    .label = `async`, `gen` or `async gen` function

passes_rustc_layout_scalar_valid_range_arg =
    expected exactly one integer literal argument

passes_rustc_layout_scalar_valid_range_not_struct =
    attribute should be applied to a struct
    .label = not a struct

passes_rustc_legacy_const_generics_index =
    #[rustc_legacy_const_generics] must have one index for each generic parameter
    .label = generic parameters

passes_rustc_legacy_const_generics_index_exceed =
    index exceeds number of arguments
    .label = there {$arg_count ->
        [one] is
        *[other] are
    } only {$arg_count} {$arg_count ->
        [one] argument
        *[other] arguments
    }

passes_rustc_legacy_const_generics_index_negative =
    arguments should be non-negative integers

passes_rustc_legacy_const_generics_only =
    #[rustc_legacy_const_generics] functions must only have const generics
    .label = non-const generic parameter

passes_rustc_lint_opt_deny_field_access =
    `#[rustc_lint_opt_deny_field_access]` should be applied to a field
    .label = not a field

passes_rustc_lint_opt_ty =
    `#[rustc_lint_opt_ty]` should be applied to a struct
    .label = not a struct

passes_rustc_pub_transparent =
    attribute should be applied to `#[repr(transparent)]` types
    .label = not a `#[repr(transparent)]` type

passes_rustc_std_internal_symbol =
    attribute should be applied to functions or statics
    .label = not a function or static

passes_should_be_applied_to_fn =
    attribute should be applied to a function definition
    .label = {$on_crate ->
        [true] cannot be applied to crates
        *[false] not a function definition
    }

passes_should_be_applied_to_static =
    attribute should be applied to a static
    .label = not a static

passes_should_be_applied_to_struct_enum =
    attribute should be applied to a struct or enum
    .label = not a struct or enum

passes_should_be_applied_to_trait =
    attribute should be applied to a trait
    .label = not a trait

passes_stability_promotable =
    attribute cannot be applied to an expression

passes_string_interpolation_only_works = string interpolation only works in `format!` invocations

passes_target_feature_on_statement =
    {passes_should_be_applied_to_fn}
    .warn = {-passes_previously_accepted}
    .label = {passes_should_be_applied_to_fn.label}

passes_trait_impl_const_stability_mismatch = const stability on the impl does not match the const stability on the trait
passes_trait_impl_const_stability_mismatch_impl_stable = this impl is (implicitly) stable...
passes_trait_impl_const_stability_mismatch_impl_unstable = this impl is unstable...
passes_trait_impl_const_stability_mismatch_trait_stable = ...but the trait is stable
passes_trait_impl_const_stability_mismatch_trait_unstable = ...but the trait is unstable

passes_trait_impl_const_stable =
    trait implementations cannot be const stable yet
    .note = see issue #67792 <https://github.com/rust-lang/rust/issues/67792> for more information

passes_transparent_incompatible =
    transparent {$target} cannot have other repr hints

passes_unexportable_adt_with_private_fields = ADT types with private fields are not exportable
    .note = `{$field_name}` is private

passes_unexportable_fn_abi = only functions with "C" ABI are exportable

passes_unexportable_generic_fn = generic functions are not exportable

passes_unexportable_item = {$descr}'s are not exportable

passes_unexportable_priv_item = private items are not exportable
    .note = is only usable at visibility `{$vis_descr}`

passes_unexportable_type_in_interface = {$desc} with `#[export_stable]` attribute uses type `{$ty}`, which is not exportable
    .label = not exportable

passes_unexportable_type_repr = types with unstable layout are not exportable

passes_unknown_external_lang_item =
    unknown external lang item: `{$lang_item}`

passes_unknown_feature =
    unknown feature `{$feature}`

passes_unknown_feature_alias =
    feature `{$alias}` has been renamed to `{$feature}`

passes_unknown_lang_item =
    definition of an unknown lang item: `{$name}`
    .label = definition of unknown lang item `{$name}`

passes_unnecessary_partial_stable_feature = the feature `{$feature}` has been partially stabilized since {$since} and is succeeded by the feature `{$implies}`
    .suggestion = if you are using features which are still unstable, change to using `{$implies}`
    .suggestion_remove = if you are using features which are now stable, remove this line

passes_unnecessary_stable_feature = the feature `{$feature}` has been stable since {$since} and no longer requires an attribute to enable

passes_unreachable_due_to_uninhabited = unreachable {$descr}
    .label = unreachable {$descr}
    .label_orig = any code following this expression is unreachable
    .note = this expression has type `{$ty}`, which is uninhabited

passes_unrecognized_argument =
    unrecognized argument

passes_unstable_attr_for_already_stable_feature =
    can't mark as unstable using an already stable feature
    .label = this feature is already stable
    .item = the stability attribute annotates this item
    .help = consider removing the attribute

passes_unsupported_attributes_in_where =
    most attributes are not supported in `where` clauses
    .help = only `#[cfg]` and `#[cfg_attr]` are supported

passes_unused =
    unused attribute
    .suggestion = remove this attribute

passes_unused_assign = value assigned to `{$name}` is never read
    .help = maybe it is overwritten before being read?

passes_unused_assign_passed = value passed to `{$name}` is never read
    .help = maybe it is overwritten before being read?

passes_unused_assign_suggestion =
    you might have meant to mutate the pointed at value being passed in, instead of changing the reference in the local binding

passes_unused_capture_maybe_capture_ref = value captured by `{$name}` is never read
    .help = did you mean to capture by reference instead?

passes_unused_default_method_body_const_note =
    `default_method_body_is_const` has been replaced with `#[const_trait]` on traits

passes_unused_duplicate =
    unused attribute
    .suggestion = remove this attribute
    .note = attribute also specified here
    .warn = {-passes_previously_accepted}

passes_unused_empty_lints_note =
    attribute `{$name}` with an empty list has no effect

passes_unused_linker_messages_note =
    the `linker_messages` lint can only be controlled at the root of a crate that needs to be linked

passes_unused_multiple =
    multiple `{$name}` attributes
    .suggestion = remove this attribute
    .note = attribute also specified here

passes_unused_no_lints_note =
    attribute `{$name}` without any lints has no effect

passes_unused_var_assigned_only = variable `{$name}` is assigned to, but never used
    .note = consider using `_{$name}` instead

passes_unused_var_maybe_capture_ref = unused variable: `{$name}`
    .help = did you mean to capture by reference instead?

passes_unused_var_remove_field = unused variable: `{$name}`
passes_unused_var_remove_field_suggestion = try removing the field

passes_unused_variable_args_in_macro = `{$name}` is captured in macro and introduced a unused variable

passes_unused_variable_try_ignore = unused variable: `{$name}`
    .suggestion = try ignoring the field

passes_unused_variable_try_prefix = unused variable: `{$name}`
    .label = unused variable
    .suggestion = if this is intentional, prefix it with an underscore


passes_used_compiler_linker =
    `used(compiler)` and `used(linker)` can't be used together

passes_used_static =
    attribute must be applied to a `static` variable
    .label = but this is a {$target}

passes_useless_assignment =
    useless assignment of {$is_field_assign ->
        [true] field
        *[false] variable
    } of type `{$ty}` to itself

passes_useless_stability =
    this stability annotation is useless
    .label = useless stability annotation
    .item = the stability attribute annotates this item
