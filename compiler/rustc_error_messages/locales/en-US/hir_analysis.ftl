hir_analysis_unrecognized_atomic_operation =
    unrecognized atomic operation function: `{$op}`
    .label = unrecognized atomic operation

hir_analysis_wrong_number_of_generic_arguments_to_intrinsic =
    intrinsic has wrong number of {$descr} parameters: found {$found}, expected {$expected}
    .label = expected {$expected} {$descr} {$expected ->
        [one] parameter
        *[other] parameters
    }

hir_analysis_unrecognized_intrinsic_function =
    unrecognized intrinsic function: `{$name}`
    .label = unrecognized intrinsic

hir_analysis_lifetimes_or_bounds_mismatch_on_trait =
    lifetime parameters or bounds on {$item_kind} `{$ident}` do not match the trait declaration
    .label = lifetimes do not match {$item_kind} in trait
    .generics_label = lifetimes in impl do not match this {$item_kind} in trait

hir_analysis_drop_impl_on_wrong_item =
    the `Drop` trait may only be implemented for local structs, enums, and unions
    .label = must be a struct, enum, or union in the current crate

hir_analysis_field_already_declared =
    field `{$field_name}` is already declared
    .label = field already declared
    .previous_decl_label = `{$field_name}` first declared here

hir_analysis_copy_impl_on_type_with_dtor =
    the trait `Copy` may not be implemented for this type; the type has a destructor
    .label = `Copy` not allowed on types with destructors

hir_analysis_multiple_relaxed_default_bounds =
    type parameter has more than one relaxed default bound, only one is supported

hir_analysis_copy_impl_on_non_adt =
    the trait `Copy` may not be implemented for this type
    .label = type is not a structure or enumeration

hir_analysis_trait_object_declared_with_no_traits =
    at least one trait is required for an object type
    .alias_span = this alias does not contain a trait

hir_analysis_ambiguous_lifetime_bound =
    ambiguous lifetime bound, explicit lifetime bound required

hir_analysis_assoc_type_binding_not_allowed =
    associated type bindings are not allowed here
    .label = associated type not allowed here

hir_analysis_typeof_reserved_keyword_used =
    `typeof` is a reserved keyword but unimplemented
    .suggestion = consider replacing `typeof(...)` with an actual type
    .label = reserved keyword

hir_analysis_value_of_associated_struct_already_specified =
    the value of the associated type `{$item_name}` (from trait `{$def_path}`) is already specified
    .label = re-bound here
    .previous_bound_label = `{$item_name}` bound here first

hir_analysis_unconstrained_opaque_type = unconstrained opaque type
    .note = `{$name}` must be used in combination with a concrete type within the same {$what}

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

hir_analysis_manual_implementation =
    manual implementations of `{$trait_name}` are experimental
    .label = manual implementations of `{$trait_name}` are experimental
    .help = add `#![feature(unboxed_closures)]` to the crate attributes to enable

hir_analysis_substs_on_overridden_impl = could not resolve substs on overridden impl

hir_analysis_unused_extern_crate =
    unused extern crate
    .suggestion = remove it

hir_analysis_extern_crate_not_idiomatic =
    `extern crate` is not idiomatic in the new edition
    .suggestion = convert it to a `{$msg_code}`

hir_analysis_expected_used_symbol = expected `used`, `used(compiler)` or `used(linker)`

hir_analysis_const_impl_for_non_const_trait =
    const `impl` for trait `{$trait_name}` which is not marked with `#[const_trait]`
    .suggestion = mark `{$trait_name}` as const
    .note = marking a trait with `#[const_trait]` ensures all default method bodies are `const`
    .adding = adding a non-const method body in the future would be a breaking change

hir_analysis_const_bound_for_non_const_trait =
    ~const can only be applied to `#[const_trait]` traits

hir_analysis_self_in_impl_self =
    `Self` is not valid in the self type of an impl block
    .note = replace `Self` with a different type

hir_analysis_linkage_type =
    must have type `*const T` or `*mut T` due to `#[linkage]` attribute
