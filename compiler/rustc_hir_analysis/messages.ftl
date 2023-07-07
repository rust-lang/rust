hir_analysis_ambiguous_lifetime_bound =
    ambiguous lifetime bound, explicit lifetime bound required

hir_analysis_assoc_type_binding_not_allowed =
    associated type bindings are not allowed here
    .label = associated type not allowed here

hir_analysis_associated_type_trait_uninferred_generic_params = cannot use the associated type of a trait with uninferred generic parameters
    .suggestion = use a fully qualified path with inferred lifetimes

hir_analysis_associated_type_trait_uninferred_generic_params_multipart_suggestion = use a fully qualified path with explicit lifetimes

hir_analysis_async_trait_impl_should_be_async =
    method `{$method_name}` should be async because the method from the trait is async
    .trait_item_label = required because the trait method is async

hir_analysis_auto_deref_reached_recursion_limit = reached the recursion limit while auto-dereferencing `{$ty}`
    .label = deref recursion limit reached
    .help = consider increasing the recursion limit by adding a `#![recursion_limit = "{$suggested_limit}"]` attribute to your crate (`{$crate_name}`)

hir_analysis_cannot_capture_late_bound_const_in_anon_const =
    cannot capture late-bound const parameter in a constant
    .label = parameter defined here

hir_analysis_cannot_capture_late_bound_ty_in_anon_const =
    cannot capture late-bound type parameter in a constant
    .label = parameter defined here

hir_analysis_cast_thin_pointer_to_fat_pointer = cannot cast thin pointer `{$expr_ty}` to fat pointer `{$cast_ty}`

hir_analysis_closure_implicit_hrtb = implicit types in closure signatures are forbidden when `for<...>` is present
    .label = `for<...>` is here

hir_analysis_const_bound_for_non_const_trait =
    ~const can only be applied to `#[const_trait]` traits

hir_analysis_const_impl_for_non_const_trait =
    const `impl` for trait `{$trait_name}` which is not marked with `#[const_trait]`
    .suggestion = mark `{$trait_name}` as const
    .note = marking a trait with `#[const_trait]` ensures all default method bodies are `const`
    .adding = adding a non-const method body in the future would be a breaking change

hir_analysis_const_param_ty_impl_on_non_adt =
    the trait `ConstParamTy` may not be implemented for this type
    .label = type is not a structure or enumeration

hir_analysis_const_specialize = cannot specialize on const impl with non-const impl

hir_analysis_copy_impl_on_non_adt =
    the trait `Copy` cannot be implemented for this type
    .label = type is not a structure or enumeration

hir_analysis_copy_impl_on_type_with_dtor =
    the trait `Copy` cannot be implemented for this type; the type has a destructor
    .label = `Copy` not allowed on types with destructors

hir_analysis_drop_impl_negative = negative `Drop` impls are not supported

hir_analysis_drop_impl_on_wrong_item =
    the `Drop` trait may only be implemented for local structs, enums, and unions
    .label = must be a struct, enum, or union in the current crate

hir_analysis_drop_impl_reservation = reservation `Drop` impls are not supported

hir_analysis_empty_specialization = specialization impl does not specialize any associated items
    .note = impl is a specialization of this impl

hir_analysis_enum_discriminant_overflowed = enum discriminant overflowed
    .label = overflowed on value after {$discr}
    .note = explicitly set `{$item_name} = {$wrapped_discr}` if that is desired outcome

hir_analysis_expected_used_symbol = expected `used`, `used(compiler)` or `used(linker)`

hir_analysis_field_already_declared =
    field `{$field_name}` is already declared
    .label = field already declared
    .previous_decl_label = `{$field_name}` first declared here

hir_analysis_function_not_found_in_trait = function not found in this trait

hir_analysis_function_not_have_default_implementation = function doesn't have a default implementation
    .note = required by this annotation

hir_analysis_functions_names_duplicated = functions names are duplicated
    .note = all `#[rustc_must_implement_one_of]` arguments must be unique

hir_analysis_impl_not_marked_default = `{$ident}` specializes an item from a parent `impl`, but that item is not marked `default`
    .label = cannot specialize default item `{$ident}`
    .ok_label = parent `impl` is here
    .note = to specialize, `{$ident}` in the parent `impl` must be marked `default`

hir_analysis_impl_not_marked_default_err = `{$ident}` specializes an item from a parent `impl`, but that item is not marked `default`
    .note = parent implementation is in crate `{$cname}`

hir_analysis_invalid_union_field =
    field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union
    .note = union fields must not have drop side-effects, which is currently enforced via either `Copy` or `ManuallyDrop<...>`

hir_analysis_invalid_union_field_sugg =
    wrap the field type in `ManuallyDrop<...>`

hir_analysis_late_bound_const_in_apit = `impl Trait` can only mention const parameters from an fn or impl
    .label = const parameter declared here

hir_analysis_late_bound_lifetime_in_apit = `impl Trait` can only mention lifetimes from an fn or impl
    .label = lifetime declared here

hir_analysis_late_bound_type_in_apit = `impl Trait` can only mention type parameters from an fn or impl
    .label = type parameter declared here

hir_analysis_lifetimes_or_bounds_mismatch_on_trait =
    lifetime parameters or bounds on {$item_kind} `{$ident}` do not match the trait declaration
    .label = lifetimes do not match {$item_kind} in trait
    .generics_label = lifetimes in impl do not match this {$item_kind} in trait
    .where_label = this `where` clause might not match the one in the trait
    .bounds_label = this bound might be missing in the impl

hir_analysis_linkage_type =
    invalid type for variable with `#[linkage]` attribute

hir_analysis_main_function_async = `main` function is not allowed to be `async`
    .label = `main` function is not allowed to be `async`

hir_analysis_main_function_generic_parameters = `main` function is not allowed to have generic parameters
    .label = `main` cannot have generic parameters

hir_analysis_main_function_return_type_generic = `main` function return type is not allowed to have generic parameters

hir_analysis_manual_implementation =
    manual implementations of `{$trait_name}` are experimental
    .label = manual implementations of `{$trait_name}` are experimental
    .help = add `#![feature(unboxed_closures)]` to the crate attributes to enable

hir_analysis_missing_one_of_trait_item = not all trait items implemented, missing one of: `{$missing_items_msg}`
    .label = missing one of `{$missing_items_msg}` in implementation
    .note = required because of this annotation

hir_analysis_missing_tilde_const = missing `~const` qualifier for specialization

hir_analysis_missing_trait_item = not all trait items implemented, missing: `{$missing_items_msg}`
    .label = missing `{$missing_items_msg}` in implementation

hir_analysis_missing_trait_item_label = `{$item}` from trait

hir_analysis_missing_trait_item_suggestion = implement the missing item: `{$snippet}`

hir_analysis_missing_trait_item_unstable = not all trait items implemented, missing: `{$missing_item_name}`
    .note = default implementation of `{$missing_item_name}` is unstable
    .some_note = use of unstable library feature '{$feature}': {$reason}
    .none_note = use of unstable library feature '{$feature}'

hir_analysis_missing_type_params =
    the type {$parameterCount ->
        [one] parameter
        *[other] parameters
    } {$parameters} must be explicitly specified
    .label = type {$parameterCount ->
        [one] parameter
        *[other] parameters
    } {$parameters} must be specified for this
    .suggestion = set the type {$parameterCount ->
        [one] parameter
        *[other] parameters
    } to the desired {$parameterCount ->
        [one] type
        *[other] types
    }
    .no_suggestion_label = missing {$parameterCount ->
        [one] reference
        *[other] references
    } to {$parameters}
    .note = because of the default `Self` reference, type parameters must be specified on object types

hir_analysis_multiple_relaxed_default_bounds =
    type parameter has more than one relaxed default bound, only one is supported

hir_analysis_must_be_name_of_associated_function = must be a name of an associated function

hir_analysis_must_implement_not_function = not a function

hir_analysis_must_implement_not_function_note = all `#[rustc_must_implement_one_of]` arguments must be associated function names

hir_analysis_must_implement_not_function_span_note = required by this annotation

hir_analysis_must_implement_one_of_attribute = the `#[rustc_must_implement_one_of]` attribute must be used with at least 2 args

hir_analysis_paren_sugar_attribute = the `#[rustc_paren_sugar]` attribute is a temporary means of controlling which traits can use parenthetical notation
    .help = add `#![feature(unboxed_closures)]` to the crate attributes to use it

hir_analysis_parenthesized_fn_trait_expansion =
    parenthesized trait syntax expands to `{$expanded_type}`

hir_analysis_pass_to_variadic_function = can't pass `{$ty}` to variadic function
    .suggestion = cast the value to `{$cast_ty}`
    .help = cast the value to `{$cast_ty}`

hir_analysis_placeholder_not_allowed_item_signatures = the placeholder `_` is not allowed within types on item signatures for {$kind}
    .label = not allowed in type signatures

hir_analysis_return_type_notation_conflicting_bound =
    ambiguous associated function `{$assoc_name}` for `{$ty_name}`
    .note = `{$assoc_name}` is declared in two supertraits: `{$first_bound}` and `{$second_bound}`

hir_analysis_return_type_notation_equality_bound =
    return type notation is not allowed to use type equality

hir_analysis_return_type_notation_illegal_param_const =
    return type notation is not allowed for functions that have const parameters
    .label = const parameter declared here
hir_analysis_return_type_notation_illegal_param_type =
    return type notation is not allowed for functions that have type parameters
    .label = type parameter declared here

hir_analysis_return_type_notation_missing_method =
    cannot find associated function `{$assoc_name}` for `{$ty_name}`

hir_analysis_return_type_notation_on_non_rpitit =
    return type notation used on function that is not `async` and does not return `impl Trait`
    .note = function returns `{$ty}`, which is not compatible with associated type return bounds
    .label = this function must be `async` or return `impl Trait`

hir_analysis_self_in_impl_self =
    `Self` is not valid in the self type of an impl block
    .note = replace `Self` with a different type

hir_analysis_simd_ffi_highly_experimental = use of SIMD type{$snip} in FFI is highly experimental and may result in invalid code
    .help = add `#![feature(simd_ffi)]` to the crate attributes to enable

hir_analysis_specialization_trait = implementing `rustc_specialization_trait` traits is unstable
    .help = add `#![feature(min_specialization)]` to the crate attributes to enable

hir_analysis_start_function_parameters = start function is not allowed to have type parameters
    .label = start function cannot have type parameters

hir_analysis_start_function_where = start function is not allowed to have a `where` clause
    .label = start function cannot have a `where` clause

hir_analysis_start_not_async = `start` is not allowed to be `async`
    .label = `start` is not allowed to be `async`

hir_analysis_start_not_target_feature = `start` is not allowed to have `#[target_feature]`
    .label = `start` is not allowed to have `#[target_feature]`

hir_analysis_start_not_track_caller = `start` is not allowed to be `#[track_caller]`
    .label = `start` is not allowed to be `#[track_caller]`

hir_analysis_static_specialize = cannot specialize on `'static` lifetime

hir_analysis_substs_on_overridden_impl = could not resolve substs on overridden impl

hir_analysis_target_feature_on_main = `main` function is not allowed to have `#[target_feature]`

hir_analysis_too_large_static = extern static is too large for the current architecture

hir_analysis_track_caller_on_main = `main` function is not allowed to be `#[track_caller]`
    .suggestion = remove this annotation

hir_analysis_trait_object_declared_with_no_traits =
    at least one trait is required for an object type
    .alias_span = this alias does not contain a trait

hir_analysis_transparent_enum_variant = transparent enum needs exactly one variant, but has {$number}
    .label = needs exactly one variant, but has {$number}
    .many_label = too many variants in `{$path}`
    .multi_label = variant here

hir_analysis_transparent_non_zero_sized = transparent {$desc} needs at most one non-zero-sized field, but has {$field_count}
    .label = needs at most one non-zero-sized field, but has {$field_count}
    .labels = this field is non-zero-sized

hir_analysis_transparent_non_zero_sized_enum = the variant of a transparent {$desc} needs at most one non-zero-sized field, but has {$field_count}
    .label = needs at most one non-zero-sized field, but has {$field_count}
    .labels = this field is non-zero-sized

hir_analysis_typeof_reserved_keyword_used =
    `typeof` is a reserved keyword but unimplemented
    .suggestion = consider replacing `typeof(...)` with an actual type
    .label = reserved keyword

hir_analysis_unconstrained_opaque_type = unconstrained opaque type
    .note = `{$name}` must be used in combination with a concrete type within the same {$what}

hir_analysis_unrecognized_atomic_operation =
    unrecognized atomic operation function: `{$op}`
    .label = unrecognized atomic operation

hir_analysis_unrecognized_intrinsic_function =
    unrecognized intrinsic function: `{$name}`
    .label = unrecognized intrinsic

hir_analysis_unused_associated_type_bounds =
    unnecessary associated type bound for not object safe associated type
    .note = this associated type has a `where Self: Sized` bound. Thus, while the associated type can be specified, it cannot be used in any way, because trait objects are not `Sized`.
    .suggestion = remove this bound

hir_analysis_value_of_associated_struct_already_specified =
    the value of the associated type `{$item_name}` (from trait `{$def_path}`) is already specified
    .label = re-bound here
    .previous_bound_label = `{$item_name}` bound here first

hir_analysis_variadic_function_compatible_convention = C-variadic function must have a compatible calling convention, like {$conventions}
    .label = C-variadic function must have a compatible calling convention

hir_analysis_variances_of = {$variances_of}

hir_analysis_where_clause_on_main = `main` function is not allowed to have a `where` clause
    .label = `main` cannot have a `where` clause

hir_analysis_wrong_number_of_generic_arguments_to_intrinsic =
    intrinsic has wrong number of {$descr} parameters: found {$found}, expected {$expected}
    .label = expected {$expected} {$descr} {$expected ->
        [one] parameter
        *[other] parameters
    }
