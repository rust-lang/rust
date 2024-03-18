hir_analysis_ambiguous_assoc_item = ambiguous associated {$assoc_kind} `{$assoc_name}` in bounds of `{$ty_param_name}`
    .label = ambiguous associated {$assoc_kind} `{$assoc_name}`

hir_analysis_ambiguous_lifetime_bound =
    ambiguous lifetime bound, explicit lifetime bound required

hir_analysis_assoc_item_not_found = associated {$assoc_kind} `{$assoc_name}` not found for `{$ty_param_name}`

hir_analysis_assoc_item_not_found_found_in_other_trait_label = there is {$identically_named ->
        [true] an
        *[false] a similarly named
    } associated {$assoc_kind} `{$suggested_name}` in the trait `{$trait_name}`
hir_analysis_assoc_item_not_found_label = associated {$assoc_kind} `{$assoc_name}` not found
hir_analysis_assoc_item_not_found_other_sugg = `{$ty_param_name}` has the following associated {$assoc_kind}
hir_analysis_assoc_item_not_found_similar_in_other_trait_sugg = change the associated {$assoc_kind} name to use `{$suggested_name}` from `{$trait_name}`
hir_analysis_assoc_item_not_found_similar_in_other_trait_with_bound_sugg = and also change the associated {$assoc_kind} name
hir_analysis_assoc_item_not_found_similar_sugg = there is an associated {$assoc_kind} with a similar name

hir_analysis_assoc_kind_mismatch = expected {$expected}, found {$got}
    .label = unexpected {$got}
    .expected_because_label = expected a {$expected} because of this associated {$expected}
    .note = the associated {$assoc_kind} is defined here
    .bound_on_assoc_const_label = bounds are not allowed on associated constants

hir_analysis_assoc_kind_mismatch_wrap_in_braces_sugg = consider adding braces here

hir_analysis_assoc_type_binding_not_allowed =
    associated type bindings are not allowed here
    .label = associated type not allowed here

hir_analysis_associated_type_trait_uninferred_generic_params = cannot use the associated type of a trait with uninferred generic parameters
    .suggestion = use a fully qualified path with inferred lifetimes

hir_analysis_associated_type_trait_uninferred_generic_params_multipart_suggestion = use a fully qualified path with explicit lifetimes

hir_analysis_auto_deref_reached_recursion_limit = reached the recursion limit while auto-dereferencing `{$ty}`
    .label = deref recursion limit reached
    .help = consider increasing the recursion limit by adding a `#![recursion_limit = "{$suggested_limit}"]` attribute to your crate (`{$crate_name}`)

hir_analysis_cannot_capture_late_bound_const =
    cannot capture late-bound const parameter in {$what}
    .label = parameter defined here

hir_analysis_cannot_capture_late_bound_lifetime =
    cannot capture late-bound lifetime in {$what}
    .label = lifetime defined here

hir_analysis_cannot_capture_late_bound_ty =
    cannot capture late-bound type parameter in {$what}
    .label = parameter defined here

hir_analysis_cast_thin_pointer_to_fat_pointer = cannot cast thin pointer `{$expr_ty}` to fat pointer `{$cast_ty}`

hir_analysis_closure_implicit_hrtb = implicit types in closure signatures are forbidden when `for<...>` is present
    .label = `for<...>` is here

hir_analysis_coerce_unsized_may = the trait `{$trait_name}` may only be implemented for a coercion between structures

hir_analysis_coerce_unsized_multi = implementing the trait `CoerceUnsized` requires multiple coercions
    .note = `CoerceUnsized` may only be implemented for a coercion between structures with one field being coerced
    .coercions_note = currently, {$number} fields need coercions: {$coercions}
    .label = requires multiple coercions

hir_analysis_coercion_between_struct_same_note = expected coercion between the same definition; expected `{$source_path}`, found `{$target_path}`

hir_analysis_coercion_between_struct_single_note = expected a single field to be coerced, none found

hir_analysis_const_bound_for_non_const_trait =
    `{$modifier}` can only be applied to `#[const_trait]` traits

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

hir_analysis_cross_crate_traits = cross-crate traits with a default impl, like `{$traits}`, can only be implemented for a struct/enum type, not `{$self_ty}`
    .label = can't implement cross-crate trait with a default impl for non-struct/enum type

hir_analysis_cross_crate_traits_defined = cross-crate traits with a default impl, like `{$traits}`, can only be implemented for a struct/enum type defined in the current crate
    .label = can't implement cross-crate trait for type in another crate

hir_analysis_dispatch_from_dyn_multi = implementing the `DispatchFromDyn` trait requires multiple coercions
    .note = the trait `DispatchFromDyn` may only be implemented for a coercion between structures with a single field being coerced
    .coercions_note = currently, {$number} fields need coercions: {$coercions}

hir_analysis_dispatch_from_dyn_repr = structs implementing `DispatchFromDyn` may not have `#[repr(packed)]` or `#[repr(C)]`

hir_analysis_dispatch_from_dyn_zst = the trait `DispatchFromDyn` may only be implemented for structs containing the field being coerced, ZST fields with 1 byte alignment, and nothing else
    .note = extra field `{$name}` of type `{$ty}` is not allowed

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

hir_analysis_escaping_bound_var_in_ty_of_assoc_const_binding =
    the type of the associated constant `{$assoc_const}` cannot capture late-bound generic parameters
    .label = its type cannot capture the late-bound {$var_def_kind} `{$var_name}`
    .var_defined_here_label = the late-bound {$var_def_kind} `{$var_name}` is defined here

hir_analysis_field_already_declared =
    field `{$field_name}` is already declared
    .label = field already declared
    .previous_decl_label = `{$field_name}` first declared here

hir_analysis_field_already_declared_both_nested =
    field `{$field_name}` is already declared
    .label = field `{$field_name}` declared in this unnamed field
    .nested_field_decl_note = field `{$field_name}` declared here
    .previous_decl_label = `{$field_name}` first declared here in this unnamed field
    .previous_nested_field_decl_note = field `{$field_name}` first declared here

hir_analysis_field_already_declared_current_nested =
    field `{$field_name}` is already declared
    .label = field `{$field_name}` declared in this unnamed field
    .nested_field_decl_note = field `{$field_name}` declared here
    .previous_decl_label = `{$field_name}` first declared here

hir_analysis_field_already_declared_nested_help =
    fields from the type of this unnamed field are considered fields of the outer type

hir_analysis_field_already_declared_previous_nested =
    field `{$field_name}` is already declared
    .label = field already declared
    .previous_decl_label = `{$field_name}` first declared here in this unnamed field
    .previous_nested_field_decl_note = field `{$field_name}` first declared here

hir_analysis_function_not_found_in_trait = function not found in this trait

hir_analysis_function_not_have_default_implementation = function doesn't have a default implementation
    .note = required by this annotation

hir_analysis_functions_names_duplicated = functions names are duplicated
    .note = all `#[rustc_must_implement_one_of]` arguments must be unique

hir_analysis_generic_args_on_overridden_impl = could not resolve generic parameters on overridden impl

hir_analysis_impl_not_marked_default = `{$ident}` specializes an item from a parent `impl`, but that item is not marked `default`
    .label = cannot specialize default item `{$ident}`
    .ok_label = parent `impl` is here
    .note = to specialize, `{$ident}` in the parent `impl` must be marked `default`

hir_analysis_impl_not_marked_default_err = `{$ident}` specializes an item from a parent `impl`, but that item is not marked `default`
    .note = parent implementation is in crate `{$cname}`

hir_analysis_inherent_dyn = cannot define inherent `impl` for a dyn auto trait
    .label = impl requires at least one non-auto trait
    .note = define and implement a new trait or type instead

hir_analysis_inherent_nominal = no nominal type found for inherent implementation
    .label = impl requires a nominal type
    .note = either implement a trait on it or create a newtype to wrap it instead
hir_analysis_inherent_primitive_ty = cannot define inherent `impl` for primitive types
    .help = consider using an extension trait instead

hir_analysis_inherent_primitive_ty_note = you could also try moving the reference to uses of `{$subty}` (such as `self`) within the implementation

hir_analysis_inherent_ty_outside = cannot define inherent `impl` for a type outside of the crate where the type is defined
    .help = consider moving this inherent impl into the crate defining the type if possible
    .span_help = alternatively add `#[rustc_has_incoherent_inherent_impls]` to the type and `#[rustc_allow_incoherent_impl]` to the relevant impl items

hir_analysis_inherent_ty_outside_new = cannot define inherent `impl` for a type outside of the crate where the type is defined
    .label = impl for type defined outside of crate.
    .note = define and implement a trait or new type instead

hir_analysis_inherent_ty_outside_primitive = cannot define inherent `impl` for primitive types outside of `core`
    .help = consider moving this inherent impl into `core` if possible
    .span_help = alternatively add `#[rustc_allow_incoherent_impl]` to the relevant impl items

hir_analysis_inherent_ty_outside_relevant = cannot define inherent `impl` for a type outside of the crate where the type is defined
    .help = consider moving this inherent impl into the crate defining the type if possible
    .span_help = alternatively add `#[rustc_allow_incoherent_impl]` to the relevant impl items

hir_analysis_invalid_union_field =
    field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union
    .note = union fields must not have drop side-effects, which is currently enforced via either `Copy` or `ManuallyDrop<...>`

hir_analysis_invalid_union_field_sugg =
    wrap the field type in `ManuallyDrop<...>`

hir_analysis_invalid_unnamed_field_ty = unnamed fields can only have struct or union types

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

hir_analysis_method_should_return_future = method should be `async` or return a future, but it is synchronous
    .note = this method is `async` so it expects a future to be returned

hir_analysis_missing_one_of_trait_item = not all trait items implemented, missing one of: `{$missing_items_msg}`
    .label = missing one of `{$missing_items_msg}` in implementation
    .note = required because of this annotation

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

hir_analysis_not_supported_delegation =
    {$descr} is not supported yet
    .label = callee defined here

hir_analysis_only_current_traits_arbitrary = only traits defined in the current crate can be implemented for arbitrary types

hir_analysis_only_current_traits_foreign = this is not defined in the current crate because this is a foreign trait

hir_analysis_only_current_traits_label = impl doesn't use only types from inside the current crate

hir_analysis_only_current_traits_name = this is not defined in the current crate because {$name} are always foreign

hir_analysis_only_current_traits_note = define and implement a trait or new type instead

hir_analysis_only_current_traits_opaque = type alias impl trait is treated as if it were foreign, because its hidden type could be from a foreign crate

hir_analysis_only_current_traits_outside = only traits defined in the current crate can be implemented for types defined outside of the crate

hir_analysis_only_current_traits_pointer = `{$pointer}` is not defined in the current crate because raw pointers are always foreign

hir_analysis_only_current_traits_pointer_sugg = consider introducing a new wrapper type

hir_analysis_only_current_traits_primitive = only traits defined in the current crate can be implemented for primitive types

hir_analysis_only_current_traits_ty = `{$ty}` is not defined in the current crate

hir_analysis_opaque_captures_higher_ranked_lifetime = `impl Trait` cannot capture {$bad_place}
    .label = `impl Trait` implicitly captures all lifetimes in scope
    .note = lifetime declared here

hir_analysis_param_in_ty_of_assoc_const_binding =
    the type of the associated constant `{$assoc_const}` must not depend on {$param_category ->
        [self] `Self`
        [synthetic] `impl Trait`
        *[normal] generic parameters
    }
    .label = its type must not depend on {$param_category ->
        [self] `Self`
        [synthetic] `impl Trait`
        *[normal] the {$param_def_kind} `{$param_name}`
    }
    .param_defined_here_label = {$param_category ->
        [synthetic] the `impl Trait` is specified here
        *[normal] the {$param_def_kind} `{$param_name}` is defined here
    }

hir_analysis_paren_sugar_attribute = the `#[rustc_paren_sugar]` attribute is a temporary means of controlling which traits can use parenthetical notation
    .help = add `#![feature(unboxed_closures)]` to the crate attributes to use it

hir_analysis_parenthesized_fn_trait_expansion =
    parenthesized trait syntax expands to `{$expanded_type}`

hir_analysis_pass_to_variadic_function = can't pass `{$ty}` to variadic function
    .suggestion = cast the value to `{$cast_ty}`
    .help = cast the value to `{$cast_ty}`

hir_analysis_placeholder_not_allowed_item_signatures = the placeholder `_` is not allowed within types on item signatures for {$kind}
    .label = not allowed in type signatures

hir_analysis_requires_note = the `{$trait_name}` impl for `{$ty}` requires that `{$error_predicate}`

hir_analysis_return_type_notation_equality_bound =
    return type notation is not allowed to use type equality

hir_analysis_return_type_notation_illegal_param_const =
    return type notation is not allowed for functions that have const parameters
    .label = const parameter declared here
hir_analysis_return_type_notation_illegal_param_type =
    return type notation is not allowed for functions that have type parameters
    .label = type parameter declared here

hir_analysis_return_type_notation_on_non_rpitit =
    return type notation used on function that is not `async` and does not return `impl Trait`
    .note = function returns `{$ty}`, which is not compatible with associated type return bounds
    .label = this function must be `async` or return `impl Trait`

hir_analysis_rpitit_refined = impl trait in impl method signature does not match trait method signature
    .suggestion = replace the return type so that it matches the trait
    .label = return type from trait method defined here
    .unmatched_bound_label = this bound is stronger than that defined on the trait
    .note = add `#[allow(refining_impl_trait)]` if it is intended for this to be part of the public API of this crate
    .feedback_note = we are soliciting feedback, see issue #121718 <https://github.com/rust-lang/rust/issues/121718> for more information

hir_analysis_self_in_impl_self =
    `Self` is not valid in the self type of an impl block
    .note = replace `Self` with a different type

hir_analysis_simd_ffi_highly_experimental = use of SIMD type{$snip} in FFI is highly experimental and may result in invalid code
    .help = add `#![feature(simd_ffi)]` to the crate attributes to enable

hir_analysis_specialization_trait = implementing `rustc_specialization_trait` traits is unstable
    .help = add `#![feature(min_specialization)]` to the crate attributes to enable

hir_analysis_start_function_parameters = `#[start]` function is not allowed to have type parameters
    .label = `#[start]` function cannot have type parameters

hir_analysis_start_function_where = `#[start]` function is not allowed to have a `where` clause
    .label = `#[start]` function cannot have a `where` clause

hir_analysis_start_not_async = `#[start]` function is not allowed to be `async`
    .label = `#[start]` is not allowed to be `async`

hir_analysis_start_not_target_feature = `#[start]` function is not allowed to have `#[target_feature]`
    .label = `#[start]` function is not allowed to have `#[target_feature]`

hir_analysis_start_not_track_caller = `#[start]` function is not allowed to be `#[track_caller]`
    .label = `#[start]` function is not allowed to be `#[track_caller]`

hir_analysis_static_mut_ref = creating a {$shared} reference to a mutable static
    .label = {$shared} reference to mutable static
    .note = {$shared ->
        [shared] this shared reference has lifetime `'static`, but if the static ever gets mutated, or a mutable reference is created, then any further use of this shared reference is Undefined Behavior
        *[mutable] this mutable reference has lifetime `'static`, but if the static gets accessed (read or written) by any other means, or any other reference is created, then any further use of this mutable reference is Undefined Behavior
    }
    .suggestion = use `addr_of!` instead to create a raw pointer
    .suggestion_mut = use `addr_of_mut!` instead to create a raw pointer

hir_analysis_static_mut_refs_lint = creating a {$shared} reference to mutable static is discouraged
    .label = {$shared} reference to mutable static
    .suggestion = use `addr_of!` instead to create a raw pointer
    .suggestion_mut = use `addr_of_mut!` instead to create a raw pointer
    .note = this will be a hard error in the 2024 edition
    .why_note = {$shared ->
        [shared] this shared reference has lifetime `'static`, but if the static ever gets mutated, or a mutable reference is created, then any further use of this shared reference is Undefined Behavior
        *[mutable] this mutable reference has lifetime `'static`, but if the static gets accessed (read or written) by any other means, or any other reference is created, then any further use of this mutable reference is Undefined Behavior
    }

hir_analysis_static_specialize = cannot specialize on `'static` lifetime

hir_analysis_tait_forward_compat = item constrains opaque type that is not in its signature
    .note = this item must mention the opaque type in its signature in order to be able to register hidden types

hir_analysis_target_feature_on_main = `main` function is not allowed to have `#[target_feature]`

hir_analysis_too_large_static = extern static is too large for the current architecture

hir_analysis_track_caller_on_main = `main` function is not allowed to be `#[track_caller]`
    .suggestion = remove this annotation

hir_analysis_trait_cannot_impl_for_ty = the trait `{$trait_name}` cannot be implemented for this type
    .label = this field does not implement `{$trait_name}`

hir_analysis_trait_object_declared_with_no_traits =
    at least one trait is required for an object type
    .alias_span = this alias does not contain a trait

hir_analysis_traits_with_defualt_impl = traits with a default impl, like `{$traits}`, cannot be implemented for {$problematic_kind} `{$self_ty}`
    .note = a trait object implements `{$traits}` if and only if `{$traits}` is one of the trait object's trait bounds

hir_analysis_transparent_enum_variant = transparent enum needs exactly one variant, but has {$number}
    .label = needs exactly one variant, but has {$number}
    .many_label = too many variants in `{$path}`
    .multi_label = variant here

hir_analysis_transparent_non_zero_sized = transparent {$desc} needs at most one field with non-trivial size or alignment, but has {$field_count}
    .label = needs at most one field with non-trivial size or alignment, but has {$field_count}
    .labels = this field has non-zero size or requires alignment

hir_analysis_transparent_non_zero_sized_enum = the variant of a transparent {$desc} needs at most one field with non-trivial size or alignment, but has {$field_count}
    .label = needs at most one field with non-trivial size or alignment, but has {$field_count}
    .labels = this field has non-zero size or requires alignment

hir_analysis_ty_of_assoc_const_binding_note = `{$assoc_const}` has type `{$ty}`

hir_analysis_ty_param_first_local = type parameter `{$param_ty}` must be covered by another type when it appears before the first local type (`{$local_type}`)
    .label = type parameter `{$param_ty}` must be covered by another type when it appears before the first local type (`{$local_type}`)
    .note = implementing a foreign trait is only possible if at least one of the types for which it is implemented is local, and no uncovered type parameters appear before that first local type
    .case_note = in this case, 'before' refers to the following order: `impl<..> ForeignTrait<T1, ..., Tn> for T0`, where `T0` is the first and `Tn` is the last

hir_analysis_ty_param_some = type parameter `{$param_ty}` must be used as the type parameter for some local type (e.g., `MyStruct<{$param_ty}>`)
    .label = type parameter `{$param_ty}` must be used as the type parameter for some local type
    .note = implementing a foreign trait is only possible if at least one of the types for which it is implemented is local
    .only_note = only traits defined in the current crate can be implemented for a type parameter

hir_analysis_type_of = {$type_of}

hir_analysis_typeof_reserved_keyword_used =
    `typeof` is a reserved keyword but unimplemented
    .suggestion = consider replacing `typeof(...)` with an actual type
    .label = reserved keyword

hir_analysis_unconstrained_opaque_type = unconstrained opaque type
    .note = `{$name}` must be used in combination with a concrete type within the same {$what}

hir_analysis_unnamed_fields_repr_field_defined = unnamed field defined here

hir_analysis_unnamed_fields_repr_field_missing_repr_c =
    named type of unnamed field must have `#[repr(C)]` representation
    .label = unnamed field defined here
    .field_ty_label = `{$field_ty}` defined here
    .suggestion = add `#[repr(C)]` to this {$field_adt_kind}

hir_analysis_unnamed_fields_repr_missing_repr_c =
    {$adt_kind} with unnamed fields must have `#[repr(C)]` representation
    .label = {$adt_kind} `{$adt_name}` defined here
    .suggestion = add `#[repr(C)]` to this {$adt_kind}

hir_analysis_unrecognized_atomic_operation =
    unrecognized atomic operation function: `{$op}`
    .label = unrecognized atomic operation

hir_analysis_unrecognized_intrinsic_function =
    unrecognized intrinsic function: `{$name}`
    .label = unrecognized intrinsic
    .help = if you're adding an intrinsic, be sure to update `check_intrinsic_type`

hir_analysis_unused_associated_type_bounds =
    unnecessary associated type bound for not object safe associated type
    .note = this associated type has a `where Self: Sized` bound. Thus, while the associated type can be specified, it cannot be used in any way, because trait objects are not `Sized`.
    .suggestion = remove this bound

hir_analysis_unused_generic_parameter =
    {$param_def_kind} `{$param_name}` is never used
    .label = unused {$param_def_kind}
    .const_param_help = if you intended `{$param_name}` to be a const parameter, use `const {$param_name}: /* Type */` instead
hir_analysis_unused_generic_parameter_adt_help =
    consider removing `{$param_name}`, referring to it in a field, or using a marker such as `{$phantom_data}`
hir_analysis_unused_generic_parameter_adt_no_phantom_data_help =
    consider removing `{$param_name}` or referring to it in a field
hir_analysis_unused_generic_parameter_ty_alias_help =
    consider removing `{$param_name}` or referring to it in the body of the type alias

hir_analysis_value_of_associated_struct_already_specified =
    the value of the associated type `{$item_name}` in trait `{$def_path}` is already specified
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
