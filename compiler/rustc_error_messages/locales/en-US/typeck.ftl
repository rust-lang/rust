typeck_field_multiply_specified_in_initializer =
    field `{$ident}` specified more than once
    .label = used more than once
    .previous_use_label = first use of `{$ident}`

typeck_unrecognized_atomic_operation =
    unrecognized atomic operation function: `{$op}`
    .label = unrecognized atomic operation

typeck_wrong_number_of_generic_arguments_to_intrinsic =
    intrinsic has wrong number of {$descr} parameters: found {$found}, expected {$expected}
    .label = expected {$expected} {$descr} {$expected ->
        [one] parameter
        *[other] parameters
    }

typeck_unrecognized_intrinsic_function =
    unrecognized intrinsic function: `{$name}`
    .label = unrecognized intrinsic

typeck_lifetimes_or_bounds_mismatch_on_trait =
    lifetime parameters or bounds on {$item_kind} `{$ident}` do not match the trait declaration
    .label = lifetimes do not match {$item_kind} in trait
    .generics_label = lifetimes in impl do not match this {$item_kind} in trait

typeck_drop_impl_on_wrong_item =
    the `Drop` trait may only be implemented for local structs, enums, and unions
    .label = must be a struct, enum, or union in the current crate

typeck_field_already_declared =
    field `{$field_name}` is already declared
    .label = field already declared
    .previous_decl_label = `{$field_name}` first declared here

typeck_copy_impl_on_type_with_dtor =
    the trait `Copy` may not be implemented for this type; the type has a destructor
    .label = `Copy` not allowed on types with destructors

typeck_multiple_relaxed_default_bounds =
    type parameter has more than one relaxed default bound, only one is supported

typeck_copy_impl_on_non_adt =
    the trait `Copy` may not be implemented for this type
    .label = type is not a structure or enumeration

typeck_trait_object_declared_with_no_traits =
    at least one trait is required for an object type
    .alias_span = this alias does not contain a trait

typeck_ambiguous_lifetime_bound =
    ambiguous lifetime bound, explicit lifetime bound required

typeck_assoc_type_binding_not_allowed =
    associated type bindings are not allowed here
    .label = associated type not allowed here

typeck_functional_record_update_on_non_struct =
    functional record update syntax requires a struct

typeck_typeof_reserved_keyword_used =
    `typeof` is a reserved keyword but unimplemented
    .suggestion = consider replacing `typeof(...)` with an actual type
    .label = reserved keyword

typeck_return_stmt_outside_of_fn_body =
    return statement outside of function body
    .encl_body_label = the return is part of this body...
    .encl_fn_label = ...not the enclosing function body

typeck_yield_expr_outside_of_generator =
    yield expression outside of generator literal

typeck_struct_expr_non_exhaustive =
    cannot create non-exhaustive {$what} using struct expression

typeck_method_call_on_unknown_type =
    the type of this value must be known to call a method on a raw pointer on it

typeck_value_of_associated_struct_already_specified =
    the value of the associated type `{$item_name}` (from trait `{$def_path}`) is already specified
    .label = re-bound here
    .previous_bound_label = `{$item_name}` bound here first

typeck_address_of_temporary_taken = cannot take address of a temporary
    .label = temporary value

typeck_add_return_type_add = try adding a return type

typeck_add_return_type_missing_here = a return type might be missing here

typeck_expected_default_return_type = expected `()` because of default return type

typeck_expected_return_type = expected `{$expected}` because of return type

typeck_unconstrained_opaque_type = unconstrained opaque type
    .note = `{$name}` must be used in combination with a concrete type within the same module

typeck_missing_type_params =
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

typeck_manual_implementation =
    manual implementations of `{$trait_name}` are experimental
    .label = manual implementations of `{$trait_name}` are experimental
    .help = add `#![feature(unboxed_closures)]` to the crate attributes to enable

typeck_substs_on_overridden_impl = could not resolve substs on overridden impl

typeck_unused_extern_crate =
    unused extern crate
    .suggestion = remove it

typeck_extern_crate_not_idiomatic =
    `extern crate` is not idiomatic in the new edition
    .suggestion = convert it to a `{$msg_code}`

typeck_safe_trait_implemented_as_unsafe =
    implementing the trait `{$trait_name}` is not unsafe

typeck_unsafe_trait_implemented_without_unsafe_keyword =
    the trait `{$trait_name}` requires an `unsafe impl` declaration

typeck_attribute_requires_unsafe_keyword =
    requires an `unsafe impl` declaration due to `#[{$attr_name}]` attribute

typeck_explicit_use_of_destructor =
    explicit use of destructor method
    .label = explicit destructor calls not allowed
    .suggestion = consider using `drop` function

typeck_unable_to_find_overloaded_call_trait =
    failed to find an overloaded call trait for closure call
    .help = make sure the `fn`/`fn_mut`/`fn_once` lang items are defined
    and have associated `call`/`call_mut`/`call_once` functions

typeck_type_parameter_not_constrained_for_impl =
    the {$kind} parameter `{$name}` is not constrained by the impl trait, self type, or predicates
    .label = unconstrained {$kind} parameter
    .first_note = expressions using a const parameter must map each value to a distinct output value
    .second_note = proving the result of expressions other than the parameter are unique is not supported

typeck_associated_items_not_distinct =
    duplicate definitions with name `{$ident}`:
    .label = duplicate definition
    .prev_def_label = previous definition of `{$ident}` here

typeck_associated_items_not_defined_in_trait =
    associated type `{$assoc_name}` not found for `{$ty_param_name}`
    .suggest_similarily_named_type = there is an associated type with a similar name
    .label_similarily_named_type = there is a similarly named associated type `{$suggested_name}` in the trait `{$trait_name}`
    .label_type_not_found = associated type `{$assoc_name}` not found

typeck_enum_discriminant_overflow =
    enum discriminant overflowed
    .label = overflowed on value after {$last_good_discriminant}
    .note = explicitly set `{$overflown_discriminant} = {$wrapped_value}` if that is desired outcome

typeck_rustc_paren_sugar_not_enabled =
    the `#[rustc_paren_sugar]` attribute is a temporary means of controlling which traits can use parenthetical notation
    .help = add `#![feature(unboxed_closures)]` to the crate attributes to use it

typeck_attribute_on_non_foreign_function =
    `#[{$attr_name}]` may only be used on foreign functions

typeck_ffi_const_and_ffi_pure_on_same_function =
    `#[ffi_const]` function cannot be `#[ffi_pure]`

typeck_cmse_nonsecure_entry_requires_c_abi =
    `#[cmse_nonsecure_entry]` requires C ABI

typeck_cmse_nonsecure_entry_requires_trust_zone_m_ext =
    `#[cmse_nonsecure_entry]` is only valid for targets with the TrustZone-M extension

typeck_track_caller_requires_cabi =
    `#[track_caller]` requires Rust ABI

typeck_export_name_contains_null_characters =
    `export_name` may not contain null characters

typeck_instruction_set_unsupported_on_target =
    target does not support `#[instruction_set]`

typeck_varargs_on_non_cabi_function =
    C-variadic function must have C or cdecl calling convention
    .label = C-variadics require C or cdecl calling convention

typeck_generic_params_on_main_function =
    `main` function is not allowed to have generic parameters
    .label = `main` cannot have generic parameters

typeck_when_clause_on_main_function =
    `main` function is not allowed to have a `where` clause
    .label = `main` cannot have a `where` clause

typeck_async_main_function =
    `main` function is not allowed to be `async`
    .label = `main` function is not allowed to be `async`

typeck_generic_return_type_on_main =
    `main` function return type is not allowed to have generic parameters

typeck_type_parameter_on_start_function =
    start function is not allowed to have type parameters
    .label = start function cannot have type parameters

typeck_where_clause_on_start_function =
    start function is not allowed to have a `where` clause
    .label = start function cannot have a `where` clause

typeck_async_start_function =
    `start` is not allowed to be `async`
    .label = `start` is not allowed to be `async`

typeck_ambiguous_associated_type =
    ambiguous associated type
    .fix_std_module_text = you are looking for the module in `std`, not the primitive type
    .fix_use_fully_qualified_syntax = use fully-qualified syntax

typeck_enum_variant_not_found =
    no variant named `{$assoc_ident}` found for enum `{$self_type}`
    .fix_similar_type = there is a variant with a similar name
    .info_label = variant not found in `{$self_type}`
    .info_label_at_enum = variant `{$assoc_ident}` not found here

typeck_expected_used_symbol = expected `used`, `used(compiler)` or `used(linker)`

typeck_invalid_dispatch_from_dyn_types_differ_too_much =
    the trait `DispatchFromDyn` may only be implemented for a coercion between structures with the same definition; expected `{$source_path}`, found `{$target_path}`

typeck_invalid_dispatch_from_dyn_invalid_repr =
    structs implementing `DispatchFromDyn` may not have `#[repr(packed)]` or `#[repr(C)]`

typeck_invalid_dispatch_from_dyn_invalid_fields =
    the trait `DispatchFromDyn` may only be implemented for structs containing the field being coerced, ZST fields with 1 byte alignment, and nothing else
    .note = extra field `{$field_name}` of type `{$ty_a}` is not allowed

typeck_invalid_dispatch_from_dyn_no_coerced_fields =
    the trait `DispatchFromDyn` may only be implemented for a coercion between structures with a single field being coerced, none found

typeck_invalid_dispatch_from_dyn_too_many_coerced_fields =
    implementing the `DispatchFromDyn` trait requires multiple coercions
    .note = the trait `DispatchFromDyn` may only be implemented for a coercion between structures with a single field being coerced
    .fields_that_need_coercions_fields = currently, {$coerced_fields_len} fields need coercions: {$coerced_fields}

typeck_invalid_dispatch_from_dyn_not_a_struct =
    the trait `DispatchFromDyn` may only be implemented for a coercion between structures

typeck_coerce_unsized_invalid_definition =
    the trait `CoerceUnsized` may only be implemented for a coercion between structures with the same definition; expected `{$source_path}`, found `{$target_path}

typeck_coerce_unsized_no_coerced_field =
    the trait `CoerceUnsized` may only be implemented for a coercion between structures with one field being coerced, none found
    .note = `CoerceUnsized` may only be implemented for a coercion between structures with one field being coerced
    .fields_that_need_coercions_fields = currently, {$coerced_fields_len} fields need coercions: {$coerced_fields}
    .label = requires multiple coercions

typeck_coerce_unsized_not_a_struct =
    the trait `CoerceUnsized` may only be implemented for a coercion between structures

typeck_cannot_implement_primitives =
    cannot define inherent `impl` for a type outside of the crate where the type is defined

typeck_explicit_impl_of_internal_structs =
    explicit impls for the `{$trait_name}` trait are not permitted
    .label = impl of `{$trait_name}` not allowed

typeck_marker_trait_impl_contains_items =
    impls for marker traits cannot contain items

typeck_type_automatically_implements_trait =
    the object type `{$object_type}` automatically implements the trait `{$trait_path}`
    .label = `{$object_type}` automatically implements trait `{$trait_path}`

typeck_cross_crate_opt_out_trait_impl_on_invalid_target =
    { $error_type ->
        [cross_crate] cross-crate traits with a default impl, like `{$trait_path}`, can only be implemented for a struct/enum type defined in the current crate
        *[invalid_type] cross-crate traits with a default impl, like `{$trait_path}`, can only be implemented for a struct/enum type, not `{$self_type}`
    }
    .label = { $error_type ->
                [cross_crate] can't implement cross-crate trait for type in another crate
                *[invalid_type] can't implement cross-crate trait with a default impl for non-struct/enum type
            }
