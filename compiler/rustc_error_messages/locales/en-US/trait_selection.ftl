trait_selection_dump_vtable_entries = vtable entries for `{$trait_ref}`: {$entries}

trait_selection_unable_to_construct_constant_value = unable to construct a constant value for the unevaluated constant {$unevaluated}

trait_selection_auto_deref_reached_recursion_limit = reached the recursion limit while auto-dereferencing `{$ty}`
    .label = deref recursion limit reached
    .help = consider increasing the recursion limit by adding a `#![recursion_limit = "{$suggested_limit}"]` attribute to your crate (`{$crate_name}`)

trait_selection_empty_on_clause_in_rustc_on_unimplemented = empty `on`-clause in `#[rustc_on_unimplemented]`
    .label = empty on-clause here

trait_selection_invalid_on_clause_in_rustc_on_unimplemented = invalid `on`-clause in `#[rustc_on_unimplemented]`
    .label = invalid on-clause here

trait_selection_no_value_in_rustc_on_unimplemented = this attribute must have a valid value
    .label = expected value here
    .note = eg `#[rustc_on_unimplemented(message="foo")]`

trait_selection_negative_positive_conflict = found both positive and negative implementation of trait `{$trait_desc}`{$self_desc ->
        [none] {""}
       *[default] {" "}for type `{$self_desc}`
    }:
    .negative_implementation_here = negative implementation here
    .negative_implementation_in_crate = negative implementation in crate `{$negative_impl_cname}`
    .positive_implementation_here = positive implementation here
    .positive_implementation_in_crate = positive implementation in crate `{$positive_impl_cname}`

trait_selection_note_access_through_trait_impl =
    {$kind}s cannot be accessed directly on a `trait`, they can only be accessed through a specific `impl`

trait_selection_note_implemented_for_other =
    `{$trait_path}` is implemented for `{$suggested_ty}`, but not for `{$original_ty}`

trait_selection_note_obligation_assignment_lhs_sized =
    the left-hand-side of an assignment must have a statically known size

trait_selection_note_obligation_binding_obligation = required by a bound in `{$item_name}`

trait_selection_note_obligation_builtin_derived_obligation =
    required because it appears within the type `{$ty}`

trait_selection_note_obligation_builtin_derived_obligation_closure =
    required because it's used within this closure

trait_selection_note_obligation_builtin_derived_obligation_generator =
    required because it's used within this {$kind}

trait_selection_note_obligation_builtin_derived_obligation_generator_witness =
    required because it captures the following types: {$captured_types}

trait_selection_note_obligation_coercion = required by cast to type `{$target}`

trait_selection_note_obligation_compare_impl_item_obligation =
    the requirement `{$predicate}` appears on the `impl`'s {$kind} `{$item_name}`
    but not on the corresponding trait's {$kind}

trait_selection_note_obligation_const_pattern_structural =
    constants used for pattern-matching must derive `PartialEq` and `Eq`

trait_selection_note_obligation_const_sized =
    constant expressions must have a statically known size

trait_selection_note_obligation_field_sized_enum =
    no field of an enum variant may have a dynamically sized type

trait_selection_note_obligation_field_sized_help =
    change the field's type to have a statically known size

trait_selection_note_obligation_field_sized_struct =
    only the last field of a struct may have a dynamically sized type

trait_selection_note_obligation_field_sized_struct_last =
    the last field of a packed struct may only have a
    dynamically sized type if it does not need drop to be run

trait_selection_note_obligation_field_sized_suggest_borrowed =
    borrowed types always have a statically known size

trait_selection_note_obligation_field_sized_union =
    no field of a union may have a dynamically sized type

trait_selection_note_obligation_function_argument_obligation_required_by =
    required by a bound introduced by this call

trait_selection_note_obligation_function_argument_obligation_tail_expr_type =
    this tail expression is of type `{$ty}`

trait_selection_note_obligation_impl_derived_obligation =
    required for `{$ty}` to implement `{$trait_path}`

trait_selection_note_obligation_impl_derived_obligation_redundant_hidden = {$count ->
    [one] {$count} redundant requirement hidden
   *[other] {$count} redundant requirements hidden
}

trait_selection_note_obligation_inline_asm_sized =
    all inline asm arguments must have a statically known size

trait_selection_note_obligation_object_cast_obligation =
    required for the cast from `{$concrete_ty}` to the object type `{$object_ty}`

trait_selection_note_obligation_object_type_bound =
    required so that the lifetime bound of `{$region}` for `{$object_ty}` is satisfied

trait_selection_note_obligation_opaque_return_type_label =
    return type was inferred to be `{$expr_ty}` here

trait_selection_note_obligation_projection_wf =
    required so that the projection `{$data}` is well-formed

trait_selection_note_obligation_reference_outlives_referent =
    required so that reference `{$ref_ty}` does not outlive its referent

trait_selection_note_obligation_repeat_element_copy =
    the `Copy` trait is required because this value will be copied for each element of the array

trait_selection_note_obligation_repeat_element_copy_help_const_fn =
    consider creating a new `const` item and initializing it with the result
    of the function call to be used in the repeat position, like
    `{$example_a}` and `{$example_b}`

trait_selection_note_obligation_repeat_element_copy_help_nightly_const_fn =
    create an inline `const` block, see RFC #2920
    <https://github.com/rust-lang/rfcs/pull/2920> for more information

trait_selection_note_obligation_shared_static =
    shared static variables must have a type that implements `Sync`

trait_selection_note_obligation_sized_argument_type =
    all function arguments must have a statically known size

trait_selection_note_obligation_sized_argument_type_help_nightly_unsized_fn_params =
    unsized fn params are gated as an unstable feature

trait_selection_note_obligation_sized_argument_type_suggest_borrowed =
    function arguments must have a statically known size, borrowed types
    always have a known size

trait_selection_note_obligation_sized_box_type =
    the type of a box expression must have a statically known size

trait_selection_note_obligation_sized_return_type =
    the return type of a function must have a statically known size

trait_selection_note_obligation_sized_yield_type =
    the yield type of a generator must have a statically known size

trait_selection_note_obligation_slice_or_array_elem =
    slice and array elements must have `Sized` type

trait_selection_note_obligation_struct_initializer_sized =
    structs must have a statically known size to be initialized

trait_selection_note_obligation_trivial_bound_help = see issue #48214

trait_selection_note_obligation_trivial_bound_help_nightly =
    add `#![feature(trivial_bounds)]` to the crate attributes to enable

trait_selection_note_obligation_tuple_elem =
    only the last element of a tuple may have a dynamically sized type

trait_selection_note_obligation_tuple_initializer_sized =
    tuples must have a statically known size to be initialized

trait_selection_note_obligation_variable_type_help_unsized_locals =
    unsized locals are gated as an unstable feature

trait_selection_note_obligation_variable_type_local =
    all local variables must have a statically known size

trait_selection_note_obligation_variable_type_local_expression = consider borrowing here

trait_selection_note_obligation_variable_type_param =
    function arguments must have a statically known size, borrowed types always have a known size

trait_selection_point_at_returns_when_relevant = this returned value is of type `{$ty}`

trait_selection_suggest_add_reference_to_arg = consider {$is_mut ->
       *[false] borrowing
        [true] mutably borrowing
    } here

trait_selection_suggest_add_reference_to_arg_label =
    the trait `{$trait_path}` is not implemented for `{$ty}`

trait_selection_suggest_add_reference_to_arg_note = the trait bound `{$trait_bound}` is not satisfied

trait_selection_suggest_await_before_try = consider `await`ing on the `Future`

trait_selection_suggest_borrowing_for_object_cast =
    consider borrowing the value, since `&{$self_ty}` can be coerced into `{$object_ty}`

trait_selection_suggest_change_borrow_mutability = consider changing this borrow's mutability

trait_selection_suggest_dereferencing_index = dereference this index

trait_selection_suggest_derive = consider annotating `{$self_ty}` with `{$annotation}`

trait_selection_suggest_floating_point_literal =
    consider using a floating-point literal by writing it with `.0`

trait_selection_suggest_fn_call_closure = consider calling this closure

trait_selection_suggest_fn_call_fn = consider calling this function

trait_selection_suggest_fn_call_msg =
    use parentheses to call the {$callable ->
       *[closure] closure
        [function] function
    }

trait_selection_suggest_fn_call_help = {trait_selection_suggest_fn_call_msg}: `{$snippet}`

trait_selection_suggest_fully_qualified_path = use the fully qualified path to an implementation

trait_selection_suggest_impl_trait =
    return type cannot have an unboxed trait object
    .suggestion =
        use `impl {$trait_obj}` as the return type, as all return paths are of type `{$last_ty}`,
        which implements `{$trait_obj}`
    .could_return_if_object_safe =
        if trait `{$trait_obj}` were object-safe, you could return a trait object
    .trait_obj_msg =
        for information on trait objects, see
        <https://doc.rust-lang.org/book/ch17-02-trait-objects.html#using-trait-objects-that-allow-for-values-of-different-types>
    .ret_vals_same_type =
        if all the returned values were of the same type you could use `impl {$trait_obj}` as the
        return type
    .impl_trait_msg =
        for information on `impl Trait`, see
        <https://doc.rust-lang.org/book/ch10-02-traits.html#returning-types-that-implement-traits>
    .create_enum = you can create a new `enum` with a variant for each returned type

trait_selection_suggest_new_overflow_limit =
    consider increasing the recursion limit by adding a
    `{$limit_attribute}` attribute to your crate (`{$crate_name}`)

trait_selection_suggest_remove_await = remove the `.await`

trait_selection_suggest_remove_await_label_return = this call returns `{$self_ty}`

trait_selection_suggest_remove_await_suggest_async =
    alternatively, consider making `fn {$ident}` asynchronous

trait_selection_suggest_remove_reference =
    consider removing {$remove_refs ->
        [one] the leading `&`-reference
       *[other] {$remove_refs} leading `&`-references
    }

trait_selection_suggest_semicolon_removal_label =
    this expression has type `{$ty}`, which implements `{$trait_path}`

trait_selection_suggest_semicolon_removal = remove this semicolon

trait_selection_suggestions_type_mismatch_in_args =
    type mismatch in {$argument_kind} arguments
    .label = expected due to this
    .found_label = found signature defined here
