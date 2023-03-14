-passes_previously_accepted =
    this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

-passes_see_issue =
    see issue #{$issue} <https://github.com/rust-lang/rust/issues/{$issue}> for more information

passes_incorrect_do_not_recommend_location =
    `#[do_not_recommend]` can only be placed on trait implementations

passes_outer_crate_level_attr =
    crate-level attribute should be an inner attribute: add an exclamation mark: `#![foo]`

passes_inner_crate_level_attr =
    crate-level attribute should be in the root module

passes_ignored_attr_with_macro =
    `#[{$sym}]` is ignored on struct fields, match arms and macro defs
    .warn = {-passes_previously_accepted}
    .note = {-passes_see_issue(issue: "80564")}

passes_ignored_attr =
    `#[{$sym}]` is ignored on struct fields and match arms
    .warn = {-passes_previously_accepted}
    .note = {-passes_see_issue(issue: "80564")}

passes_inline_ignored_function_prototype =
    `#[inline]` is ignored on function prototypes

passes_inline_ignored_constants =
    `#[inline]` is ignored on constants
    .warn = {-passes_previously_accepted}
    .note = {-passes_see_issue(issue: "65833")}

passes_inline_not_fn_or_closure =
    attribute should be applied to function or closure
    .label = not a function or closure

passes_no_coverage_ignored_function_prototype =
    `#[no_coverage]` is ignored on function prototypes

passes_no_coverage_propagate =
    `#[no_coverage]` does not propagate into items and must be applied to the contained functions directly

passes_no_coverage_fn_defn =
    `#[no_coverage]` may only be applied to function definitions

passes_no_coverage_not_coverable =
    `#[no_coverage]` must be applied to coverable code
    .label = not coverable code

passes_should_be_applied_to_fn =
    attribute should be applied to a function definition
    .label = {$on_crate ->
        [true] cannot be applied to crates
        *[false] not a function definition
    }

passes_naked_tracked_caller =
    cannot use `#[track_caller]` with `#[naked]`

passes_should_be_applied_to_struct_enum =
    attribute should be applied to a struct or enum
    .label = not a struct or enum

passes_should_be_applied_to_trait =
    attribute should be applied to a trait
    .label = not a trait

passes_target_feature_on_statement =
    {passes_should_be_applied_to_fn}
    .warn = {-passes_previously_accepted}
    .label = {passes_should_be_applied_to_fn.label}

passes_should_be_applied_to_static =
    attribute should be applied to a static
    .label = not a static

passes_doc_expect_str =
    doc {$attr_name} attribute expects a string: #[doc({$attr_name} = "a")]

passes_doc_alias_empty =
    {$attr_str} attribute cannot have empty value

passes_doc_alias_bad_char =
    {$char_} character isn't allowed in {$attr_str}

passes_doc_alias_start_end =
    {$attr_str} cannot start or end with ' '

passes_doc_alias_bad_location =
    {$attr_str} isn't allowed on {$location}

passes_doc_alias_not_an_alias =
    {$attr_str} is the same as the item's name

passes_doc_alias_duplicated = doc alias is duplicated
    .label = first defined here

passes_doc_alias_not_string_literal =
    `#[doc(alias("a"))]` expects string literals

passes_doc_alias_malformed =
    doc alias attribute expects a string `#[doc(alias = "a")]` or a list of strings `#[doc(alias("a", "b"))]`

passes_doc_keyword_empty_mod =
    `#[doc(keyword = "...")]` should be used on empty modules

passes_doc_keyword_not_mod =
    `#[doc(keyword = "...")]` should be used on modules

passes_doc_keyword_invalid_ident =
    `{$doc_keyword}` is not a valid identifier

passes_doc_fake_variadic_not_valid =
    `#[doc(fake_variadic)]` must be used on the first of a set of tuple or fn pointer trait impls with varying arity

passes_doc_keyword_only_impl =
    `#[doc(keyword = "...")]` should be used on impl blocks

passes_doc_inline_conflict_first =
    this attribute...

passes_doc_inline_conflict_second =
    {"."}..conflicts with this attribute

passes_doc_inline_conflict =
    conflicting doc inlining attributes
    .help = remove one of the conflicting attributes

passes_doc_inline_only_use =
    this attribute can only be applied to a `use` item
    .label = only applicable on `use` items
    .not_a_use_item_label = not a `use` item
    .note = read <https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#inline-and-no_inline> for more information

passes_doc_attr_not_crate_level =
    `#![doc({$attr_name} = "...")]` isn't allowed as a crate-level attribute

passes_attr_crate_level =
    this attribute can only be applied at the crate level
    .suggestion = to apply to the crate, use an inner attribute
    .help = to apply to the crate, use an inner attribute
    .note = read <https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#at-the-crate-level> for more information

passes_doc_test_unknown =
    unknown `doc(test)` attribute `{$path}`

passes_doc_test_takes_list =
    `#[doc(test(...)]` takes a list of attributes

passes_doc_primitive =
    `doc(primitive)` should never have been stable

passes_doc_cfg_hide_takes_list =
    `#[doc(cfg_hide(...)]` takes a list of attributes

passes_doc_test_unknown_any =
    unknown `doc` attribute `{$path}`

passes_doc_test_unknown_spotlight =
    unknown `doc` attribute `{$path}`
    .note = `doc(spotlight)` was renamed to `doc(notable_trait)`
    .suggestion = use `notable_trait` instead
    .no_op_note = `doc(spotlight)` is now a no-op

passes_doc_test_unknown_include =
    unknown `doc` attribute `{$path}`
    .suggestion = use `doc = include_str!` instead

passes_doc_invalid =
    invalid `doc` attribute

passes_pass_by_value =
    `pass_by_value` attribute should be applied to a struct, enum or type alias
    .label = is not a struct, enum or type alias

passes_allow_incoherent_impl =
    `rustc_allow_incoherent_impl` attribute should be applied to impl items.
    .label = the only currently supported targets are inherent methods

passes_has_incoherent_inherent_impl =
    `rustc_has_incoherent_inherent_impls` attribute should be applied to types or traits.
    .label = only adts, extern types and traits are supported

passes_both_ffi_const_and_pure =
    `#[ffi_const]` function cannot be `#[ffi_pure]`

passes_ffi_pure_invalid_target =
    `#[ffi_pure]` may only be used on foreign functions

passes_ffi_const_invalid_target =
    `#[ffi_const]` may only be used on foreign functions

passes_ffi_returns_twice_invalid_target =
    `#[ffi_returns_twice]` may only be used on foreign functions

passes_must_use_async =
    `must_use` attribute on `async` functions applies to the anonymous `Future` returned by the function, not the value within
    .label = this attribute does nothing, the `Future`s returned by async functions are already `must_use`

passes_must_use_no_effect =
    `#[must_use]` has no effect when applied to {$article} {$target}

passes_must_not_suspend =
    `must_not_suspend` attribute should be applied to a struct, enum, or trait
    .label = is not a struct, enum, or trait

passes_cold =
    {passes_should_be_applied_to_fn}
    .warn = {-passes_previously_accepted}
    .label = {passes_should_be_applied_to_fn.label}

passes_link =
    attribute should be applied to an `extern` block with non-Rust ABI
    .warn = {-passes_previously_accepted}
    .label = not an `extern` block

passes_link_name =
    attribute should be applied to a foreign function or static
    .warn = {-passes_previously_accepted}
    .label = not a foreign function or static
    .help = try `#[link(name = "{$value}")]` instead

passes_no_link =
    attribute should be applied to an `extern crate` item
    .label = not an `extern crate` item

passes_export_name =
    attribute should be applied to a free function, impl method or static
    .label = not a free function, impl method or static

passes_rustc_layout_scalar_valid_range_not_struct =
    attribute should be applied to a struct
    .label = not a struct

passes_rustc_layout_scalar_valid_range_arg =
    expected exactly one integer literal argument

passes_rustc_legacy_const_generics_only =
    #[rustc_legacy_const_generics] functions must only have const generics
    .label = non-const generic parameter

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

passes_rustc_dirty_clean =
    attribute requires -Z query-dep-graph to be enabled

passes_link_section =
    attribute should be applied to a function or static
    .warn = {-passes_previously_accepted}
    .label = not a function or static

passes_no_mangle_foreign =
    `#[no_mangle]` has no effect on a foreign {$foreign_item_kind}
    .warn = {-passes_previously_accepted}
    .label = foreign {$foreign_item_kind}
    .note = symbol names in extern blocks are not mangled
    .suggestion = remove this attribute

passes_no_mangle =
    attribute should be applied to a free function, impl method or static
    .warn = {-passes_previously_accepted}
    .label = not a free function, impl method or static

passes_repr_ident =
    meta item in `repr` must be an identifier

passes_repr_conflicting =
    conflicting representation hints

passes_used_static =
    attribute must be applied to a `static` variable

passes_used_compiler_linker =
    `used(compiler)` and `used(linker)` can't be used together

passes_allow_internal_unstable =
    attribute should be applied to a macro
    .label = not a macro

passes_debug_visualizer_placement =
    attribute should be applied to a module

passes_debug_visualizer_invalid =
    invalid argument
    .note_1 = expected: `natvis_file = "..."`
    .note_2 = OR
    .note_3 = expected: `gdb_script_file = "..."`

passes_debug_visualizer_unreadable =
    couldn't read {$file}: {$error}

passes_rustc_allow_const_fn_unstable =
    attribute should be applied to `const fn`
    .label = not a `const fn`

passes_rustc_std_internal_symbol =
    attribute should be applied to functions or statics
    .label = not a function or static

passes_const_trait =
    attribute should be applied to a trait

passes_stability_promotable =
    attribute cannot be applied to an expression

passes_deprecated =
    attribute is ignored here

passes_macro_use =
    `#[{$name}]` only has an effect on `extern crate` and modules

passes_macro_export =
    `#[macro_export]` only has an effect on macro definitions

passes_plugin_registrar =
    `#[plugin_registrar]` only has an effect on functions

passes_unused_empty_lints_note =
    attribute `{$name}` with an empty list has no effect

passes_unused_no_lints_note =
    attribute `{$name}` without any lints has no effect

passes_unused_default_method_body_const_note =
    `default_method_body_is_const` has been replaced with `#[const_trait]` on traits

passes_unused =
    unused attribute
    .suggestion = remove this attribute

passes_non_exported_macro_invalid_attrs =
    attribute should be applied to function or closure
    .label = not a function or closure

passes_unused_duplicate =
    unused attribute
    .suggestion = remove this attribute
    .note = attribute also specified here
    .warn = {-passes_previously_accepted}

passes_unused_multiple =
    multiple `{$name}` attributes
    .suggestion = remove this attribute
    .note = attribute also specified here

passes_rustc_lint_opt_ty =
    `#[rustc_lint_opt_ty]` should be applied to a struct
    .label = not a struct

passes_rustc_lint_opt_deny_field_access =
    `#[rustc_lint_opt_deny_field_access]` should be applied to a field
    .label = not a field

passes_link_ordinal =
    attribute should be applied to a foreign function or static
    .label = not a foreign function or static

passes_collapse_debuginfo =
    `collapse_debuginfo` attribute should be applied to macro definitions
    .label = not a macro definition

passes_deprecated_annotation_has_no_effect =
    this `#[deprecated]` annotation has no effect
    .suggestion = remove the unnecessary deprecation attribute

passes_unknown_external_lang_item =
    unknown external lang item: `{$lang_item}`

passes_missing_panic_handler =
    `#[panic_handler]` function required, but not found

passes_missing_lang_item =
    language item required, but not found: `{$name}`
    .note = this can occur when a binary crate with `#![no_std]` is compiled for a target where `{$name}` is defined in the standard library
    .help = you may be able to compile for a target that doesn't need `{$name}`, specify a target with `--target` or in `.cargo/config`

passes_lang_item_on_incorrect_target =
    `{$name}` language item must be applied to a {$expected_target}
    .label = attribute should be applied to a {$expected_target}, not a {$actual_target}

passes_unknown_lang_item =
    definition of an unknown language item: `{$name}`
    .label = definition of unknown language item `{$name}`

passes_invalid_attr_at_crate_level =
    `{$name}` attribute cannot be used at crate level
    .suggestion = perhaps you meant to use an outer attribute

passes_duplicate_diagnostic_item_in_crate =
    duplicate diagnostic item in crate `{$crate_name}`: `{$name}`.
    .note = the diagnostic item is first defined in crate `{$orig_crate_name}`.

passes_diagnostic_item_first_defined =
    the diagnostic item is first defined here

passes_abi =
    abi: {$abi}

passes_align =
    align: {$align}

passes_size =
    size: {$size}

passes_homogeneous_aggregate =
    homogeneous_aggregate: {$homogeneous_aggregate}

passes_layout_of =
    layout_of({$normalized_ty}) = {$ty_layout}

passes_unrecognized_field =
    unrecognized field name `{$name}`

passes_layout =
    layout error: {$layout_error}

passes_feature_stable_twice =
    feature `{$feature}` is declared stable since {$since}, but was previously declared stable since {$prev_since}

passes_feature_previously_declared =
    feature `{$feature}` is declared {$declared}, but was previously declared {$prev_declared}

passes_expr_not_allowed_in_context =
    {$expr} is not allowed in a `{$context}`

passes_const_impl_const_trait =
    const `impl`s must be for traits marked with `#[const_trait]`
    .note = this trait must be annotated with `#[const_trait]`

passes_break_non_loop =
    `break` with value from a `{$kind}` loop
    .label = can only break with a value inside `loop` or breakable block
    .label2 = you can't `break` with a value in a `{$kind}` loop
    .suggestion = use `break` on its own without a value inside this `{$kind}` loop
    .break_expr_suggestion = alternatively, you might have meant to use the available loop label

passes_continue_labeled_block =
    `continue` pointing to a labeled block
    .label = labeled blocks cannot be `continue`'d
    .block_label = labeled block the `continue` points to

passes_break_inside_closure =
    `{$name}` inside of a closure
    .label = cannot `{$name}` inside of a closure
    .closure_label = enclosing closure

passes_break_inside_async_block =
    `{$name}` inside of an `async` block
    .label = cannot `{$name}` inside of an `async` block
    .async_block_label = enclosing `async` block

passes_outside_loop =
    `{$name}` outside of a loop{$is_break ->
        [true] {" or labeled block"}
        *[false] {""}
    }
    .label = cannot `{$name}` outside of a loop{$is_break ->
        [true] {" or labeled block"}
        *[false] {""}
    }

passes_unlabeled_in_labeled_block =
    unlabeled `{$cf_type}` inside of a labeled block
    .label = `{$cf_type}` statements that would diverge to or through a labeled block need to bear a label

passes_unlabeled_cf_in_while_condition =
    `break` or `continue` with no label in the condition of a `while` loop
    .label = unlabeled `{$cf_type}` in the condition of a `while` loop

passes_cannot_inline_naked_function =
    naked functions cannot be inlined

passes_undefined_naked_function_abi =
    Rust ABI is unsupported in naked functions

passes_no_patterns =
    patterns not allowed in naked function parameters

passes_params_not_allowed =
    referencing function parameters is not allowed in naked functions
    .help = follow the calling convention in asm block to use parameters

passes_naked_functions_asm_block =
    naked functions must contain a single asm block
    .label_multiple_asm = multiple asm blocks are unsupported in naked functions
    .label_non_asm = non-asm is unsupported in naked functions

passes_naked_functions_operands =
    only `const` and `sym` operands are supported in naked functions

passes_naked_functions_asm_options =
    asm options unsupported in naked functions: {$unsupported_options}

passes_naked_functions_must_use_noreturn =
    asm in naked functions must use `noreturn` option
    .suggestion = consider specifying that the asm block is responsible for returning from the function

passes_attr_only_on_main =
    `{$attr}` attribute can only be used on `fn main()`

passes_attr_only_on_root_main =
    `{$attr}` attribute can only be used on root `fn main()`

passes_attr_only_in_functions =
    `{$attr}` attribute can only be used on functions

passes_multiple_rustc_main =
    multiple functions with a `#[rustc_main]` attribute
    .first = first `#[rustc_main]` function
    .additional = additional `#[rustc_main]` function

passes_multiple_start_functions =
    multiple `start` functions
    .label = multiple `start` functions
    .previous = previous `#[start]` function here

passes_extern_main =
    the `main` function cannot be declared in an `extern` block

passes_unix_sigpipe_values =
    valid values for `#[unix_sigpipe = "..."]` are `inherit`, `sig_ign`, or `sig_dfl`

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

passes_duplicate_lang_item =
    found duplicate lang item `{$lang_item_name}`
    .first_defined_span = the lang item is first defined here
    .first_defined_crate_depends = the lang item is first defined in crate `{$orig_crate_name}` (which `{$orig_dependency_of}` depends on)
    .first_defined_crate = the lang item is first defined in crate `{$orig_crate_name}`.
    .first_definition_local = first definition in the local crate (`{$orig_crate_name}`)
    .second_definition_local = second definition in the local crate (`{$crate_name}`)
    .first_definition_path = first definition in `{$orig_crate_name}` loaded from {$orig_path}
    .second_definition_path = second definition in `{$crate_name}` loaded from {$path}

passes_duplicate_lang_item_crate =
    duplicate lang item in crate `{$crate_name}`: `{$lang_item_name}`.
    .first_defined_span = the lang item is first defined here
    .first_defined_crate_depends = the lang item is first defined in crate `{$orig_crate_name}` (which `{$orig_dependency_of}` depends on)
    .first_defined_crate = the lang item is first defined in crate `{$orig_crate_name}`.
    .first_definition_local = first definition in the local crate (`{$orig_crate_name}`)
    .second_definition_local = second definition in the local crate (`{$crate_name}`)
    .first_definition_path = first definition in `{$orig_crate_name}` loaded from {$orig_path}
    .second_definition_path = second definition in `{$crate_name}` loaded from {$path}

passes_duplicate_lang_item_crate_depends =
    duplicate lang item in crate `{$crate_name}` (which `{$dependency_of}` depends on): `{$lang_item_name}`.
    .first_defined_span = the lang item is first defined here
    .first_defined_crate_depends = the lang item is first defined in crate `{$orig_crate_name}` (which `{$orig_dependency_of}` depends on)
    .first_defined_crate = the lang item is first defined in crate `{$orig_crate_name}`.
    .first_definition_local = first definition in the local crate (`{$orig_crate_name}`)
    .second_definition_local = second definition in the local crate (`{$crate_name}`)
    .first_definition_path = first definition in `{$orig_crate_name}` loaded from {$orig_path}
    .second_definition_path = second definition in `{$crate_name}` loaded from {$path}

passes_incorrect_target =
    `{$name}` language item must be applied to a {$kind} with {$at_least ->
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

passes_useless_assignment =
    useless assignment of {$is_field_assign ->
        [true] field
        *[false] variable
    } of type `{$ty}` to itself

passes_only_has_effect_on =
    `#[{$attr_name}]` only has an effect on {$target_name ->
        [function] functions
        [module] modules
        [implementation_block] implementation blocks
        *[unspecified] (unspecified--this is a compiler bug)
    }

passes_object_lifetime_err =
    {$repr}

passes_unrecognized_repr_hint =
    unrecognized representation hint
    .help = valid reprs are `C`, `align`, `packed`, `transparent`, `simd`, `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `i128`, `u128`, `isize`, `usize`

passes_attr_application_enum =
    attribute should be applied to an enum
    .label = not an enum

passes_attr_application_struct =
    attribute should be applied to a struct
    .label = not a struct

passes_attr_application_struct_union =
    attribute should be applied to a struct or union
    .label = not a struct or union

passes_attr_application_struct_enum_union =
    attribute should be applied to a struct, enum, or union
    .label = not a struct, enum, or union

passes_attr_application_struct_enum_function_union =
    attribute should be applied to a struct, enum, function, or union
    .label = not a struct, enum, function, or union

passes_transparent_incompatible =
    transparent {$target} cannot have other repr hints

passes_deprecated_attribute =
    deprecated attribute must be paired with either stable or unstable attribute

passes_useless_stability =
    this stability annotation is useless
    .label = useless stability annotation
    .item = the stability attribute annotates this item

passes_invalid_stability =
    invalid stability version found
    .label = invalid stability version
    .item = the stability attribute annotates this item

passes_cannot_stabilize_deprecated =
    an API can't be stabilized after it is deprecated
    .label = invalid version
    .item = the stability attribute annotates this item

passes_invalid_deprecation_version =
    invalid deprecation version found
    .label = invalid deprecation version
    .item = the stability attribute annotates this item

passes_missing_stability_attr =
    {$descr} has missing stability attribute

passes_missing_const_stab_attr =
    {$descr} has missing const stability attribute

passes_trait_impl_const_stable =
    trait implementations cannot be const stable yet
    .note = see issue #67792 <https://github.com/rust-lang/rust/issues/67792> for more information

passes_feature_only_on_nightly =
    `#![feature]` may not be used on the {$release_channel} release channel

passes_unknown_feature =
    unknown feature `{$feature}`

passes_implied_feature_not_exist =
    feature `{$implied_by}` implying `{$feature}` does not exist

passes_duplicate_feature_err =
    the feature `{$feature}` has already been declared

passes_missing_const_err =
    attributes `#[rustc_const_unstable]` and `#[rustc_const_stable]` require the function or method to be `const`
    .help = make the function or method const
    .label = attribute specified here

passes_dead_codes =
    { $multiple ->
      *[true] multiple {$descr}s are
       [false] { $num ->
         [one] {$descr} {$name_list} is
        *[other] {$descr}s {$name_list} are
       }
    } never {$participle}

passes_change_fields_to_be_of_unit_type =
    consider changing the { $num ->
      [one] field
     *[other] fields
    } to be of unit type to suppress this warning while preserving the field numbering, or remove the { $num ->
      [one] field
     *[other] fields
    }

passes_parent_info =
    {$num ->
      [one] {$descr}
     *[other] {$descr}s
    } in this {$parent_descr}

passes_ignored_derived_impls =
    `{$name}` has {$trait_list_len ->
      [one] a derived impl
     *[other] derived impls
    } for the {$trait_list_len ->
      [one] trait {$trait_list}, but this is
     *[other] traits {$trait_list}, but these are
    } intentionally ignored during dead code analysis

passes_proc_macro_bad_sig = {$kind} has incorrect signature

passes_skipping_const_checks = skipping const checks

passes_invalid_macro_export_arguments = `{$name}` isn't a valid `#[macro_export]` argument

passes_invalid_macro_export_arguments_too_many_items = `#[macro_export]` can only take 1 or 0 arguments
