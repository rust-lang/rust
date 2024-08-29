ast_passes_anon_struct_or_union_not_allowed =
    anonymous {$struct_or_union}s are not allowed outside of unnamed struct or union fields
    .label = anonymous {$struct_or_union} declared here

ast_passes_assoc_const_without_body =
    associated constant in `impl` without body
    .suggestion = provide a definition for the constant

ast_passes_assoc_fn_without_body =
    associated function in `impl` without body
    .suggestion = provide a definition for the function

ast_passes_assoc_type_without_body =
    associated type in `impl` without body
    .suggestion = provide a definition for the type

ast_passes_at_least_one_trait = at least one trait must be specified

ast_passes_auto_generic = auto traits cannot have generic parameters
    .label = auto trait cannot have generic parameters
    .suggestion = remove the parameters

ast_passes_auto_items = auto traits cannot have associated items
    .label = {ast_passes_auto_items}
    .suggestion = remove these associated items

ast_passes_auto_super_lifetime = auto traits cannot have super traits or lifetime bounds
    .label = {ast_passes_auto_super_lifetime}
    .suggestion = remove the super traits or lifetime bounds

ast_passes_bad_c_variadic = only foreign, `unsafe extern "C"`, or `unsafe extern "C-unwind"` functions may have a C-variadic arg

ast_passes_bare_fn_invalid_safety = function pointers cannot be declared with `safe` safety qualifier
    .suggestion = remove safe from this item

ast_passes_body_in_extern = incorrect `{$kind}` inside `extern` block
    .cannot_have = cannot have a body
    .invalid = the invalid body
    .existing = `extern` blocks define existing foreign {$kind}s and {$kind}s inside of them cannot have a body

ast_passes_bound_in_context = bounds on `type`s in {$ctx} have no effect

ast_passes_const_and_async = functions cannot be both `const` and `async`
    .const = `const` because of this
    .async = `async` because of this
    .label = {""}

ast_passes_const_and_c_variadic = functions cannot be both `const` and C-variadic
    .const = `const` because of this
    .variadic = C-variadic because of this

ast_passes_const_bound_trait_object = const trait bounds are not allowed in trait object types

ast_passes_const_without_body =
    free constant item without body
    .suggestion = provide a definition for the constant

ast_passes_constraint_on_negative_bound =
    associated type constraints not allowed on negative bounds

ast_passes_equality_in_where = equality constraints are not yet supported in `where` clauses
    .label = not supported
    .suggestion = if `{$ident}` is an associated type you're trying to set, use the associated type binding syntax
    .suggestion_path = if `{$trait_segment}::{$potential_assoc}` is an associated type you're trying to set, use the associated type binding syntax
    .note = see issue #20041 <https://github.com/rust-lang/rust/issues/20041> for more information

ast_passes_extern_block_suggestion = if you meant to declare an externally defined function, use an `extern` block

ast_passes_extern_fn_qualifiers = functions in `extern` blocks cannot have qualifiers
    .label = in this `extern` block
    .suggestion = remove this qualifier

ast_passes_extern_invalid_safety = items in unadorned `extern` blocks cannot have safety qualifiers
    .suggestion = add unsafe to this `extern` block

ast_passes_extern_item_ascii = items in `extern` blocks cannot use non-ascii identifiers
    .label = in this `extern` block
    .note = this limitation may be lifted in the future; see issue #83942 <https://github.com/rust-lang/rust/issues/83942> for more information

ast_passes_extern_keyword_link = for more information, visit https://doc.rust-lang.org/std/keyword.extern.html

ast_passes_extern_types_cannot = `type`s inside `extern` blocks cannot have {$descr}
    .suggestion = remove the {$remove_descr}
    .label = `extern` block begins here

ast_passes_feature_on_non_nightly = `#![feature]` may not be used on the {$channel} release channel
    .suggestion = remove the attribute
    .stable_since = the feature `{$name}` has been stable since `{$since}` and no longer requires an attribute to enable

ast_passes_fieldless_union = unions cannot have zero fields

ast_passes_fn_body_extern = incorrect function inside `extern` block
    .cannot_have = cannot have a body
    .suggestion = remove the invalid body
    .help = you might have meant to write a function accessible through FFI, which can be done by writing `extern fn` outside of the `extern` block
    .label = `extern` blocks define existing foreign functions and functions inside of them cannot have a body

ast_passes_fn_param_c_var_args_not_last =
    `...` must be the last argument of a C-variadic function

ast_passes_fn_param_doc_comment =
    documentation comments cannot be applied to function parameters
    .label = doc comments are not allowed here

ast_passes_fn_param_forbidden_attr =
    allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters

ast_passes_fn_param_forbidden_self =
    `self` parameter is only allowed in associated functions
    .label = not semantically valid as function parameter
    .note = associated functions are those in `impl` or `trait` definitions

ast_passes_fn_param_too_many =
    function can not have more than {$max_num_args} arguments

ast_passes_fn_without_body =
    free function without a body
    .suggestion = provide a definition for the function

ast_passes_forbidden_bound =
    bounds cannot be used in this context

ast_passes_forbidden_const_param =
    late-bound const parameters cannot be used currently

ast_passes_forbidden_default =
    `default` is only allowed on items in trait impls
    .label = `default` because of this

ast_passes_forbidden_non_lifetime_param =
    only lifetime parameters can be used in this context

ast_passes_generic_before_constraints = generic arguments must come before the first constraint
    .constraints = {$constraint_len ->
    [one] constraint
    *[other] constraints
    }
    .args = generic {$args_len ->
    [one] argument
    *[other] arguments
    }
    .empty_string = {""},
    .suggestion = move the {$constraint_len ->
    [one] constraint
    *[other] constraints
    } after the generic {$args_len ->
    [one] argument
    *[other] arguments
    }

ast_passes_generic_default_trailing = generic parameters with a default must be trailing

ast_passes_impl_trait_path = `impl Trait` is not allowed in path parameters

ast_passes_incompatible_features = `{$f1}` and `{$f2}` are incompatible, using them at the same time is not allowed
    .help = remove one of these features

ast_passes_inherent_cannot_be = inherent impls cannot be {$annotation}
    .because = {$annotation} because of this
    .type = inherent impl for this type
    .only_trait = only trait implementations may be annotated with {$annotation}

ast_passes_invalid_unnamed_field =
    unnamed fields are not allowed outside of structs or unions
    .label = unnamed field declared here

ast_passes_invalid_unnamed_field_ty =
    unnamed fields can only have struct or union types
    .label = not a struct or union

ast_passes_item_invalid_safety = items outside of `unsafe extern {"{ }"}` cannot be declared with `safe` safety qualifier
    .suggestion = remove safe from this item

ast_passes_item_underscore = `{$kind}` items in this context need a name
    .label = `_` is not a valid name for this `{$kind}` item

ast_passes_match_arm_with_no_body =
    `match` arm with no body
    .suggestion = add a body after the pattern

ast_passes_missing_unsafe_on_extern = extern blocks must be unsafe

ast_passes_module_nonascii = trying to load file for module `{$name}` with non-ascii identifier name
    .help = consider using the `#[path]` attribute to specify filesystem path

ast_passes_negative_bound_not_supported =
    negative bounds are not supported

ast_passes_negative_bound_with_parenthetical_notation =
    parenthetical notation may not be used for negative bounds

ast_passes_nested_impl_trait = nested `impl Trait` is not allowed
    .outer = outer `impl Trait`
    .inner = nested `impl Trait` here

ast_passes_nested_lifetimes = nested quantification of lifetimes

ast_passes_nomangle_ascii = `#[no_mangle]` requires ASCII identifier

ast_passes_obsolete_auto = `impl Trait for .. {"{}"}` is an obsolete syntax
    .help = use `auto trait Trait {"{}"}` instead

ast_passes_optional_trait_object = `?Trait` is not permitted in trait object types

ast_passes_optional_trait_supertrait = `?Trait` is not permitted in supertraits
    .note = traits are `?{$path_str}` by default

ast_passes_out_of_order_params = {$param_ord} parameters must be declared prior to {$max_param} parameters
    .suggestion = reorder the parameters: lifetimes, then consts and types

ast_passes_pattern_in_bodiless = patterns aren't allowed in functions without bodies
    .label = pattern not allowed in function without body

ast_passes_pattern_in_fn_pointer = patterns aren't allowed in function pointer types

ast_passes_pattern_in_foreign = patterns aren't allowed in foreign function declarations
    .label = pattern not allowed in foreign function

ast_passes_precise_capturing_duplicated = duplicate `use<...>` precise capturing syntax
    .label = second `use<...>` here

ast_passes_precise_capturing_not_allowed_here = `use<...>` precise capturing syntax not allowed in {$loc}

ast_passes_show_span = {$msg}

ast_passes_stability_outside_std = stability attributes may not be used outside of the standard library

ast_passes_static_without_body =
    free static item without body
    .suggestion = provide a definition for the static

ast_passes_tilde_const_disallowed = `~const` is not allowed here
    .closure = closures cannot have `~const` trait bounds
    .function = this function is not `const`, so it cannot have `~const` trait bounds
    .trait = this trait is not a `#[const_trait]`, so it cannot have `~const` trait bounds
    .trait_impl = this impl is not `const`, so it cannot have `~const` trait bounds
    .impl = inherent impls cannot have `~const` trait bounds
    .trait_assoc_ty = associated types in non-`#[const_trait]` traits cannot have `~const` trait bounds
    .trait_impl_assoc_ty = associated types in non-const impls cannot have `~const` trait bounds
    .inherent_assoc_ty = inherent associated types cannot have `~const` trait bounds
    .object = trait objects cannot have `~const` trait bounds
    .item = this item cannot have `~const` trait bounds

ast_passes_trait_fn_const =
    functions in {$in_impl ->
        [true] trait impls
        *[false] traits
    } cannot be declared const
    .label = functions in {$in_impl ->
        [true] trait impls
        *[false] traits
    } cannot be const
    .const_context_label = this declares all associated functions implicitly const
    .remove_const_sugg = remove the `const`{$requires_multiple_changes ->
        [true] {" ..."}
        *[false] {""}
    }
    .make_impl_const_sugg = ... and declare the impl to be const instead
    .make_trait_const_sugg = ... and declare the trait to be a `#[const_trait]` instead

ast_passes_trait_object_single_bound = only a single explicit lifetime bound is permitted

ast_passes_ty_alias_without_body =
    free type alias without body
    .suggestion = provide a definition for the type

ast_passes_unsafe_item = {$kind} cannot be declared unsafe

ast_passes_unsafe_negative_impl = negative impls cannot be unsafe
    .negative = negative because of this
    .unsafe = unsafe because of this

ast_passes_unsafe_static =
    static items cannot be declared with `unsafe` safety qualifier outside of `extern` block

ast_passes_visibility_not_permitted =
    visibility qualifiers are not permitted here
    .enum_variant = enum variants and their fields always share the visibility of the enum they are in
    .trait_impl = trait items always share the visibility of their trait
    .individual_impl_items = place qualifiers on individual impl items instead
    .individual_foreign_items = place qualifiers on individual foreign items instead
    .remove_qualifier_sugg = remove the qualifier

ast_passes_where_clause_after_type_alias = where clauses are not allowed after the type for type aliases
    .note = see issue #112792 <https://github.com/rust-lang/rust/issues/112792> for more information
    .help = add `#![feature(lazy_type_alias)]` to the crate attributes to enable

ast_passes_where_clause_before_type_alias = where clauses are not allowed before the type for type aliases
    .note = see issue #89122 <https://github.com/rust-lang/rust/issues/89122> for more information
    .remove_suggestion = remove this `where`
    .move_suggestion = move it to the end of the type declaration
