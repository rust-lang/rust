hir_analysis_abi_custom_clothed_function =
    items with the `"custom"` ABI can only be declared externally or defined via naked functions
    .suggestion = convert this to an `#[unsafe(naked)]` function

hir_analysis_ambiguous_assoc_item = ambiguous associated {$assoc_kind} `{$assoc_ident}` in bounds of `{$qself}`
    .label = ambiguous associated {$assoc_kind} `{$assoc_ident}`

hir_analysis_ambiguous_lifetime_bound =
    ambiguous lifetime bound, explicit lifetime bound required

hir_analysis_assoc_item_constraints_not_allowed_here =
    associated item constraints are not allowed here
    .label = associated item constraint not allowed here

hir_analysis_assoc_item_is_private = {$kind} `{$name}` is private
    .label = private {$kind}
    .defined_here_label = the {$kind} is defined here

hir_analysis_assoc_item_not_found = associated {$assoc_kind} `{$assoc_ident}` not found for `{$qself}`

hir_analysis_assoc_item_not_found_found_in_other_trait_label = there is {$identically_named ->
        [true] an
        *[false] a similarly named
    } associated {$assoc_kind} `{$suggested_name}` in the trait `{$trait_name}`
hir_analysis_assoc_item_not_found_label = associated {$assoc_kind} `{$assoc_ident}` not found
hir_analysis_assoc_item_not_found_other_sugg = `{$qself}` has the following associated {$assoc_kind}
hir_analysis_assoc_item_not_found_similar_in_other_trait_qpath_sugg =
    consider fully qualifying{$identically_named ->
        [true] {""}
        *[false] {" "}and renaming
    } the associated {$assoc_kind}
hir_analysis_assoc_item_not_found_similar_in_other_trait_sugg = change the associated {$assoc_kind} name to use `{$suggested_name}` from `{$trait_name}`
hir_analysis_assoc_item_not_found_similar_in_other_trait_with_bound_sugg = ...and changing the associated {$assoc_kind} name
hir_analysis_assoc_item_not_found_similar_sugg = there is an associated {$assoc_kind} with a similar name

hir_analysis_assoc_kind_mismatch = expected {$expected}, found {$got}
    .label = unexpected {$got}
    .expected_because_label = expected a {$expected} because of this associated {$expected}
    .note = the associated {$assoc_kind} is defined here
    .bound_on_assoc_const_label = bounds are not allowed on associated constants

hir_analysis_assoc_kind_mismatch_wrap_in_braces_sugg = consider adding braces here

hir_analysis_associated_type_trait_uninferred_generic_params = cannot use the {$what} of a trait with uninferred generic parameters
    .suggestion = use a fully qualified path with inferred lifetimes

hir_analysis_associated_type_trait_uninferred_generic_params_multipart_suggestion = use a fully qualified path with explicit lifetimes

hir_analysis_auto_deref_reached_recursion_limit = reached the recursion limit while auto-dereferencing `{$ty}`
    .label = deref recursion limit reached
    .help = consider increasing the recursion limit by adding a `#![recursion_limit = "{$suggested_limit}"]` attribute to your crate (`{$crate_name}`)

hir_analysis_bad_precise_capture = expected {$kind} parameter in `use<...>` precise captures list, found {$found}

hir_analysis_bad_return_type_notation_position = return type notation not allowed in this position yet

hir_analysis_cannot_capture_late_bound_const =
    cannot capture late-bound const parameter in {$what}
    .label = parameter defined here

hir_analysis_cannot_capture_late_bound_lifetime =
    cannot capture late-bound lifetime in {$what}
    .label = lifetime defined here

hir_analysis_cannot_capture_late_bound_ty =
    cannot capture late-bound type parameter in {$what}
    .label = parameter defined here

hir_analysis_closure_implicit_hrtb = implicit types in closure signatures are forbidden when `for<...>` is present
    .label = `for<...>` is here

hir_analysis_cmse_call_generic =
    function pointers with the `"C-cmse-nonsecure-call"` ABI cannot contain generics in their type

hir_analysis_cmse_entry_generic =
    functions with the `"C-cmse-nonsecure-entry"` ABI cannot contain generics in their type

hir_analysis_cmse_inputs_stack_spill =
    arguments for `{$abi}` function too large to pass via registers
    .label = {$plural ->
        [false] this argument doesn't
        *[true] these arguments don't
    } fit in the available registers
    .note = functions with the `{$abi}` ABI must pass all their arguments via the 4 32-bit available argument registers

hir_analysis_cmse_output_stack_spill =
    return value of `{$abi}` function too large to pass via registers
    .label = this type doesn't fit in the available registers
    .note1 = functions with the `{$abi}` ABI must pass their result via the available return registers
    .note2 = the result must either be a (transparently wrapped) i64, u64 or f64, or be at most 4 bytes in size

hir_analysis_coerce_multi = implementing `{$trait_name}` does not allow multiple fields to be coerced
    .note = the trait `{$trait_name}` may only be implemented when a single field is being coerced
    .label = these fields must be coerced for `{$trait_name}` to be valid

hir_analysis_coerce_pointee_no_field = `CoercePointee` can only be derived on `struct`s with at least one field

hir_analysis_coerce_pointee_no_user_validity_assertion = asserting applicability of `derive(CoercePointee)` on a target data is forbidden

hir_analysis_coerce_pointee_not_concrete_ty = `derive(CoercePointee)` is only applicable to `struct`

hir_analysis_coerce_pointee_not_struct = `derive(CoercePointee)` is only applicable to `struct`, instead of `{$kind}`

hir_analysis_coerce_pointee_not_transparent = `derive(CoercePointee)` is only applicable to `struct` with `repr(transparent)` layout

hir_analysis_coerce_unsized_field_validity = for `{$ty}` to have a valid implementation of `{$trait_name}`, it must be possible to coerce the field of type `{$field_ty}`
    .label = `{$field_ty}` must be a pointer, reference, or smart pointer that is allowed to be unsized

hir_analysis_coerce_unsized_may = the trait `{$trait_name}` may only be implemented for a coercion between structures

hir_analysis_coerce_zero = implementing `{$trait_name}` requires a field to be coerced

hir_analysis_coercion_between_struct_same_note = expected coercion between the same definition; expected `{$source_path}`, found `{$target_path}`

hir_analysis_coercion_between_struct_single_note = expected a single field to be coerced, none found

hir_analysis_const_bound_for_non_const_trait = `{$modifier}` can only be applied to `#[const_trait]` traits
    .label = can't be applied to `{$trait_name}`
    .note = `{$trait_name}` can't be used with `{$modifier}` because it isn't annotated with `#[const_trait]`
    .suggestion = {$suggestion_pre}mark `{$trait_name}` as `#[const_trait]` to allow it to have `const` implementations

hir_analysis_const_impl_for_non_const_trait = const `impl` for trait `{$trait_name}` which is not marked with `#[const_trait]`
    .label = this trait is not `const`
    .suggestion = {$suggestion_pre}mark `{$trait_name}` as `#[const_trait]` to allow it to have `const` implementations
    .note = marking a trait with `#[const_trait]` ensures all default method bodies are `const`
    .adding = adding a non-const method body in the future would be a breaking change

hir_analysis_const_param_ty_impl_on_non_adt =
    the trait `ConstParamTy` may not be implemented for this type
    .label = type is not a structure or enumeration

hir_analysis_const_param_ty_impl_on_unsized =
    the trait `ConstParamTy` may not be implemented for this type
    .label = type is not `Sized`

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

hir_analysis_dispatch_from_dyn_repr = structs implementing `DispatchFromDyn` may not have `#[repr(packed)]` or `#[repr(C)]`

hir_analysis_dispatch_from_dyn_zst = the trait `DispatchFromDyn` may only be implemented for structs containing the field being coerced, ZST fields with 1 byte alignment that don't mention type/const generics, and nothing else
    .note = extra field `{$name}` of type `{$ty}` is not allowed

hir_analysis_drop_impl_negative = negative `Drop` impls are not supported

hir_analysis_drop_impl_on_wrong_item =
    the `Drop` trait may only be implemented for local structs, enums, and unions
    .label = must be a struct, enum, or union in the current crate

hir_analysis_drop_impl_reservation = reservation `Drop` impls are not supported

hir_analysis_duplicate_precise_capture = cannot capture parameter `{$name}` twice
    .label = parameter captured again here

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
    .label = impl for type defined outside of crate
    .note = define and implement a trait or new type instead

hir_analysis_inherent_ty_outside_primitive = cannot define inherent `impl` for primitive types outside of `core`
    .help = consider moving this inherent impl into `core` if possible
    .span_help = alternatively add `#[rustc_allow_incoherent_impl]` to the relevant impl items

hir_analysis_inherent_ty_outside_relevant = cannot define inherent `impl` for a type outside of the crate where the type is defined
    .help = consider moving this inherent impl into the crate defining the type if possible
    .span_help = alternatively add `#[rustc_allow_incoherent_impl]` to the relevant impl items

hir_analysis_invalid_generic_receiver_ty = invalid generic `self` parameter type: `{$receiver_ty}`
    .note = type of `self` must not be a method generic parameter type

hir_analysis_invalid_generic_receiver_ty_help =
    use a concrete type such as `self`, `&self`, `&mut self`, `self: Box<Self>`, `self: Rc<Self>`, `self: Arc<Self>`, or `self: Pin<P>` (where P is one of the previous types except `Self`)

hir_analysis_invalid_receiver_ty = invalid `self` parameter type: `{$receiver_ty}`
    .note = type of `self` must be `Self` or some type implementing `Receiver`

hir_analysis_invalid_receiver_ty_help =
    consider changing to `self`, `&self`, `&mut self`, or a type implementing `Receiver` such as `self: Box<Self>`, `self: Rc<Self>`, or `self: Arc<Self>`

hir_analysis_invalid_receiver_ty_help_no_arbitrary_self_types =
    consider changing to `self`, `&self`, `&mut self`, `self: Box<Self>`, `self: Rc<Self>`, `self: Arc<Self>`, or `self: Pin<P>` (where P is one of the previous types except `Self`)

hir_analysis_invalid_receiver_ty_help_nonnull_note =
    `NonNull` does not implement `Receiver` because it has methods that may shadow the referent; consider wrapping your `NonNull` in a newtype wrapper for which you implement `Receiver`

hir_analysis_invalid_receiver_ty_help_weak_note =
    `Weak` does not implement `Receiver` because it has methods that may shadow the referent; consider wrapping your `Weak` in a newtype wrapper for which you implement `Receiver`

hir_analysis_invalid_receiver_ty_no_arbitrary_self_types = invalid `self` parameter type: `{$receiver_ty}`
    .note = type of `self` must be `Self` or a type that dereferences to it

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

hir_analysis_lifetime_implicitly_captured = `impl Trait` captures lifetime parameter, but it is not mentioned in `use<...>` precise captures list
    .param_label = all lifetime parameters originating from a trait are captured implicitly

hir_analysis_lifetime_must_be_first = lifetime parameter `{$name}` must be listed before non-lifetime parameters
    .label = move the lifetime before this parameter

hir_analysis_lifetime_not_captured = `impl Trait` captures lifetime parameter, but it is not mentioned in `use<...>` precise captures list
    .label = lifetime captured due to being mentioned in the bounds of the `impl Trait`
    .param_label = this lifetime parameter is captured

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
    .some_note = use of unstable library feature `{$feature}`: {$reason}
    .none_note = use of unstable library feature `{$feature}`

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
    .note = because the parameter {$parameterCount ->
        [one] default references
        *[other] defaults reference
    } `Self`, the {$parameterCount ->
        [one] parameter
        *[other] parameters
    } must be specified on the object type

hir_analysis_multiple_relaxed_default_bounds =
    type parameter has more than one relaxed default bound, only one is supported

hir_analysis_must_be_name_of_associated_function = must be a name of an associated function

hir_analysis_must_implement_not_function = not a function

hir_analysis_must_implement_not_function_note = all `#[rustc_must_implement_one_of]` arguments must be associated function names

hir_analysis_must_implement_not_function_span_note = required by this annotation

hir_analysis_must_implement_one_of_attribute = the `#[rustc_must_implement_one_of]` attribute must be used with at least 2 args

hir_analysis_no_variant_named = no variant named `{$ident}` found for enum `{$ty}`

hir_analysis_not_supported_delegation = {$descr}
    .label = callee defined here

hir_analysis_only_current_traits_adt = `{$name}` is not defined in the current crate

hir_analysis_only_current_traits_arbitrary = only traits defined in the current crate can be implemented for arbitrary types

hir_analysis_only_current_traits_foreign = this is not defined in the current crate because this is a foreign trait

hir_analysis_only_current_traits_name = this is not defined in the current crate because {$name} are always foreign

hir_analysis_only_current_traits_note = define and implement a trait or new type instead

hir_analysis_only_current_traits_note_more_info = for more information see https://doc.rust-lang.org/reference/items/implementations.html#orphan-rules

hir_analysis_only_current_traits_note_uncovered = impl doesn't have any local type before any uncovered type parameters

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

hir_analysis_param_not_captured = `impl Trait` must mention all {$kind} parameters in scope in `use<...>`
    .label = {$kind} parameter is implicitly captured by this `impl Trait`
    .note = currently, all {$kind} parameters are required to be mentioned in the precise captures list

hir_analysis_paren_sugar_attribute = the `#[rustc_paren_sugar]` attribute is a temporary means of controlling which traits can use parenthetical notation
    .help = add `#![feature(unboxed_closures)]` to the crate attributes to use it

hir_analysis_parenthesized_fn_trait_expansion =
    parenthesized trait syntax expands to `{$expanded_type}`

hir_analysis_placeholder_not_allowed_item_signatures = the placeholder `_` is not allowed within types on item signatures for {$kind}
    .label = not allowed in type signatures
hir_analysis_precise_capture_self_alias = `Self` can't be captured in `use<...>` precise captures list, since it is an alias
    .label = `Self` is not a generic argument, but an alias to the type of the {$what}

hir_analysis_recursive_generic_parameter = {$param_def_kind} `{$param_name}` is only used recursively
    .label = {$param_def_kind} must be used non-recursively in the definition
    .note = all type parameters must be used in a non-recursive way in order to constrain their variance

hir_analysis_redundant_lifetime_args = unnecessary lifetime parameter `{$victim}`
    .note = you can use the `{$candidate}` lifetime directly, in place of `{$victim}`

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

hir_analysis_rpitit_refined_lifetimes = impl trait in impl method captures fewer lifetimes than in trait
    .suggestion = modify the `use<..>` bound to capture the same lifetimes that the trait does
    .note = add `#[allow(refining_impl_trait)]` if it is intended for this to be part of the public API of this crate
    .feedback_note = we are soliciting feedback, see issue #121718 <https://github.com/rust-lang/rust/issues/121718> for more information

hir_analysis_self_in_impl_self =
    `Self` is not valid in the self type of an impl block
    .note = replace `Self` with a different type

hir_analysis_self_in_type_alias = `Self` is not allowed in type aliases
    .label = `Self` is only available in impls, traits, and concrete type definitions

hir_analysis_self_ty_not_captured = `impl Trait` must mention the `Self` type of the trait in `use<...>`
    .label = `Self` type parameter is implicitly captured by this `impl Trait`
    .note = currently, all type parameters are required to be mentioned in the precise captures list

hir_analysis_simd_ffi_highly_experimental = use of SIMD type{$snip} in FFI is highly experimental and may result in invalid code
    .help = add `#![feature(simd_ffi)]` to the crate attributes to enable

hir_analysis_specialization_trait = implementing `rustc_specialization_trait` traits is unstable
    .help = add `#![feature(min_specialization)]` to the crate attributes to enable

hir_analysis_static_specialize = cannot specialize on `'static` lifetime

hir_analysis_supertrait_item_multiple_shadowee = items from several supertraits are shadowed: {$traits}

hir_analysis_supertrait_item_shadowee = item from `{$supertrait}` is shadowed by a subtrait item

hir_analysis_supertrait_item_shadowing = trait item `{$item}` from `{$subtrait}` shadows identically named item from supertrait

hir_analysis_tait_forward_compat2 = item does not constrain `{$opaque_type}`
    .note = consider removing `#[define_opaque]` or adding an empty `#[define_opaque()]`
    .opaque = this opaque type is supposed to be constrained

hir_analysis_target_feature_on_main = `main` function is not allowed to have `#[target_feature]`

hir_analysis_too_large_static = extern static is too large for the target architecture

hir_analysis_track_caller_on_main = `main` function is not allowed to be `#[track_caller]`
    .suggestion = remove this annotation

hir_analysis_trait_cannot_impl_for_ty = the trait `{$trait_name}` cannot be implemented for this type
    .label = this field does not implement `{$trait_name}`

hir_analysis_trait_object_declared_with_no_traits =
    at least one trait is required for an object type
    .alias_span = this alias does not contain a trait

hir_analysis_traits_with_default_impl = traits with a default impl, like `{$traits}`, cannot be implemented for {$problematic_kind} `{$self_ty}`
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

hir_analysis_ty_param_first_local = type parameter `{$param}` must be covered by another type when it appears before the first local type (`{$local_type}`)
    .label = type parameter `{$param}` must be covered by another type when it appears before the first local type (`{$local_type}`)
    .note = implementing a foreign trait is only possible if at least one of the types for which it is implemented is local, and no uncovered type parameters appear before that first local type
    .case_note = in this case, 'before' refers to the following order: `impl<..> ForeignTrait<T1, ..., Tn> for T0`, where `T0` is the first and `Tn` is the last

hir_analysis_ty_param_some = type parameter `{$param}` must be used as the type parameter for some local type (e.g., `MyStruct<{$param}>`)
    .label = type parameter `{$param}` must be used as the type parameter for some local type
    .note = implementing a foreign trait is only possible if at least one of the types for which it is implemented is local
    .only_note = only traits defined in the current crate can be implemented for a type parameter

hir_analysis_type_of = {$ty}

hir_analysis_typeof_reserved_keyword_used =
    `typeof` is a reserved keyword but unimplemented
    .suggestion = consider replacing `typeof(...)` with an actual type
    .label = reserved keyword

hir_analysis_unconstrained_generic_parameter = the {$param_def_kind} `{$param_name}` is not constrained by the impl trait, self type, or predicates
    .label = unconstrained {$param_def_kind}
    .const_param_note = expressions using a const parameter must map each value to a distinct output value
    .const_param_note2 = proving the result of expressions other than the parameter are unique is not supported

hir_analysis_unconstrained_opaque_type = unconstrained opaque type
    .note = `{$name}` must be used in combination with a concrete type within the same {$what}

hir_analysis_unrecognized_intrinsic_function =
    unrecognized intrinsic function: `{$name}`
    .label = unrecognized intrinsic
    .help = if you're adding an intrinsic, be sure to update `check_intrinsic_type`

hir_analysis_unused_associated_type_bounds =
    unnecessary associated type bound for dyn-incompatible associated type
    .note = this associated type has a `where Self: Sized` bound, and while the associated type can be specified, it cannot be used because trait objects are never `Sized`
    .suggestion = remove this bound

hir_analysis_unused_generic_parameter =
    {$param_def_kind} `{$param_name}` is never used
    .label = unused {$param_def_kind}
    .const_param_help = if you intended `{$param_name}` to be a const parameter, use `const {$param_name}: /* Type */` instead
    .usage_spans = `{$param_name}` is named here, but is likely unused in the containing type

hir_analysis_unused_generic_parameter_adt_help =
    consider removing `{$param_name}`, referring to it in a field, or using a marker such as `{$phantom_data}`
hir_analysis_unused_generic_parameter_adt_no_phantom_data_help =
    consider removing `{$param_name}` or referring to it in a field
hir_analysis_unused_generic_parameter_ty_alias_help =
    consider removing `{$param_name}` or referring to it in the body of the type alias

hir_analysis_useless_impl_item = this item cannot be used as its where bounds are not satisfied for the `Self` type

hir_analysis_value_of_associated_struct_already_specified =
    the value of the associated type `{$item_name}` in trait `{$def_path}` is already specified
    .label = re-bound here
    .previous_bound_label = `{$item_name}` bound here first

hir_analysis_variadic_function_compatible_convention = C-variadic functions with the {$convention} calling convention are not supported
    .label = C-variadic function must have a compatible calling convention

hir_analysis_variances_of = {$variances}

hir_analysis_where_clause_on_main = `main` function is not allowed to have a `where` clause
    .label = `main` cannot have a `where` clause

hir_analysis_within_macro = due to this macro variable

hir_analysis_wrong_number_of_generic_arguments_to_intrinsic =
    intrinsic has wrong number of {$descr} parameters: found {$found}, expected {$expected}
    .label = expected {$expected} {$descr} {$expected ->
        [one] parameter
        *[other] parameters
    }
