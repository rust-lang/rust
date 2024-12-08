hir_typeck_add_missing_parentheses_in_range = you must surround the range in parentheses to call its `{$func_name}` function

hir_typeck_add_return_type_add = try adding a return type

hir_typeck_add_return_type_missing_here = a return type might be missing here

hir_typeck_address_of_temporary_taken = cannot take address of a temporary
    .label = temporary value

hir_typeck_arg_mismatch_indeterminate = argument type mismatch was detected, but rustc had trouble determining where
    .note = we would appreciate a bug report: https://github.com/rust-lang/rust/issues/new

hir_typeck_candidate_trait_note = `{$trait_name}` defines an item `{$item_name}`{$action_or_ty ->
    [NONE] {""}
    [implement] , perhaps you need to implement it
    *[other] , perhaps you need to restrict type parameter `{$action_or_ty}` with it
}

hir_typeck_cannot_cast_to_bool = cannot cast `{$expr_ty}` as `bool`
    .suggestion = compare with zero instead
    .help = compare with zero instead
    .label = unsupported cast

hir_typeck_cast_enum_drop = cannot cast enum `{$expr_ty}` into integer `{$cast_ty}` because it implements `Drop`

hir_typeck_cast_thin_pointer_to_wide_pointer = cannot cast thin pointer `{$expr_ty}` to wide pointer `{$cast_ty}`
    .teach_help = Thin pointers are "simple" pointers: they are purely a reference to a
        memory address.

        Wide pointers are pointers referencing "Dynamically Sized Types" (also
        called DST). DST don't have a statically known size, therefore they can
        only exist behind some kind of pointers that contain additional
        information. Slices and trait objects are DSTs. In the case of slices,
        the additional information the wide pointer holds is their size.

        To fix this error, don't try to cast directly between thin and wide
        pointers.

        For more information about casts, take a look at The Book:
        https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions",

hir_typeck_cast_unknown_pointer = cannot cast {$to ->
    [true] to
    *[false] from
    } a pointer of an unknown kind
    .label_to = needs more type information
    .note = the type information given here is insufficient to check whether the pointer cast is valid
    .label_from = the type information given here is insufficient to check whether the pointer cast is valid

hir_typeck_const_select_must_be_const = this argument must be a `const fn`
    .help = consult the documentation on `const_eval_select` for more information

hir_typeck_const_select_must_be_fn = this argument must be a function item
    .note = expected a function item, found {$ty}
    .help = consult the documentation on `const_eval_select` for more information

hir_typeck_convert_to_str = try converting the passed type into a `&str`

hir_typeck_convert_using_method = try using `{$sugg}` to convert `{$found}` to `{$expected}`

hir_typeck_ctor_is_private = tuple struct constructor `{$def}` is private

hir_typeck_dependency_on_unit_never_type_fallback = this function depends on never type fallback being `()`
    .note = in edition 2024, the requirement `{$obligation}` will fail
    .help = specify the types explicitly

hir_typeck_deref_is_empty = this expression `Deref`s to `{$deref_ty}` which implements `is_empty`

hir_typeck_expected_default_return_type = expected `()` because of default return type

hir_typeck_expected_return_type = expected `{$expected}` because of return type

hir_typeck_explicit_destructor = explicit use of destructor method
    .label = explicit destructor calls not allowed
    .suggestion = consider using `drop` function

hir_typeck_field_multiply_specified_in_initializer =
    field `{$ident}` specified more than once
    .label = used more than once
    .previous_use_label = first use of `{$ident}`

hir_typeck_fn_item_to_variadic_function = can't pass a function item to a variadic function
    .suggestion = use a function pointer instead
    .help = a function item is zero-sized and needs to be cast into a function pointer to be used in FFI
    .note = for more information on function items, visit https://doc.rust-lang.org/reference/types/function-item.html

hir_typeck_fru_expr = this expression does not end in a comma...
hir_typeck_fru_expr2 = ... so this is interpreted as a `..` range expression, instead of functional record update syntax
hir_typeck_fru_note = this expression may have been misinterpreted as a `..` range expression
hir_typeck_fru_suggestion =
    to set the remaining fields{$expr ->
        [NONE]{""}
        *[other] {" "}from `{$expr}`
    }, separate the last named field with a comma

hir_typeck_functional_record_update_on_non_struct =
    functional record update syntax requires a struct

hir_typeck_help_set_edition_cargo = set `edition = "{$edition}"` in `Cargo.toml`
hir_typeck_help_set_edition_standalone = pass `--edition {$edition}` to `rustc`

hir_typeck_int_to_fat = cannot cast `{$expr_ty}` to a pointer that {$known_wide ->
    [true] is
    *[false] may be
    } wide
hir_typeck_int_to_fat_label = creating a `{$cast_ty}` requires both an address and {$metadata}
hir_typeck_int_to_fat_label_nightly = consider casting this expression to `*const ()`, then using `core::ptr::from_raw_parts`

hir_typeck_invalid_callee = expected function, found {$ty}

hir_typeck_lossy_provenance_int2ptr =
    strict provenance disallows casting integer `{$expr_ty}` to pointer `{$cast_ty}`
    .suggestion = use `.with_addr()` to adjust a valid pointer in the same allocation, to this address
    .help = if you can't comply with strict provenance and don't have a pointer with the correct provenance you can use `std::ptr::with_exposed_provenance()` instead

hir_typeck_lossy_provenance_ptr2int =
    under strict provenance it is considered bad style to cast pointer `{$expr_ty}` to integer `{$cast_ty}`
    .suggestion = use `.addr()` to obtain the address of a pointer
    .help = if you can't comply with strict provenance and need to expose the pointer provenance you can use `.expose_provenance()` instead

hir_typeck_missing_parentheses_in_range = can't call method `{$method_name}` on type `{$ty_str}`

hir_typeck_never_type_fallback_flowing_into_unsafe_call = never type fallback affects this call to an `unsafe` function
    .help = specify the type explicitly
hir_typeck_never_type_fallback_flowing_into_unsafe_deref = never type fallback affects this raw pointer dereference
    .help = specify the type explicitly
hir_typeck_never_type_fallback_flowing_into_unsafe_method = never type fallback affects this call to an `unsafe` method
    .help = specify the type explicitly
hir_typeck_never_type_fallback_flowing_into_unsafe_path = never type fallback affects this `unsafe` function
    .help = specify the type explicitly
hir_typeck_never_type_fallback_flowing_into_unsafe_union_field = never type fallback affects this union access
    .help = specify the type explicitly

hir_typeck_no_associated_item = no {$item_kind} named `{$item_name}` found for {$ty_prefix} `{$ty_str}`{$trait_missing_method ->
    [true] {""}
    *[other] {" "}in the current scope
}

hir_typeck_note_caller_chooses_ty_for_ty_param = the caller chooses a type for `{$ty_param_name}` which can be different from `{$found_ty}`

hir_typeck_note_edition_guide = for more on editions, read https://doc.rust-lang.org/edition-guide

hir_typeck_option_result_asref = use `{$def_path}::as_ref` to convert `{$expected_ty}` to `{$expr_ty}`
hir_typeck_option_result_cloned = use `{$def_path}::cloned` to clone the value inside the `{$def_path}`
hir_typeck_option_result_copied = use `{$def_path}::copied` to copy the value inside the `{$def_path}`

hir_typeck_pass_to_variadic_function = can't pass `{$ty}` to variadic function
    .suggestion = cast the value to `{$cast_ty}`
    .teach_help = certain types, like `{$ty}`, must be casted before passing them to a variadic function, because of arcane ABI rules dictated by the C standard

hir_typeck_ptr_cast_add_auto_to_object = adding {$traits_len ->
    [1] an auto trait {$traits}
    *[other] auto traits {$traits}
} to a trait object in a pointer cast may cause UB later on

hir_typeck_remove_semi_for_coerce = you might have meant to return the `match` expression
hir_typeck_remove_semi_for_coerce_expr = this could be implicitly returned but it is a statement, not a tail expression
hir_typeck_remove_semi_for_coerce_ret = the `match` arms can conform to this return type
hir_typeck_remove_semi_for_coerce_semi = the `match` is a statement because of this semicolon, consider removing it
hir_typeck_remove_semi_for_coerce_suggestion = remove this semicolon

hir_typeck_return_stmt_outside_of_fn_body =
    {$statement_kind} statement outside of function body
    .encl_body_label = the {$statement_kind} is part of this body...
    .encl_fn_label = ...not the enclosing function body

hir_typeck_rpit_box_return_expr = if you change the return type to expect trait objects, box the returned expressions

hir_typeck_rpit_change_return_type = you could change the return type to be a boxed trait object

hir_typeck_rustcall_incorrect_args =
    functions with the "rust-call" ABI must take a single non-self tuple argument

hir_typeck_self_ctor_from_outer_item = can't reference `Self` constructor from outer item
    .label = the inner item doesn't inherit generics from this impl, so `Self` is invalid to reference
    .suggestion = replace `Self` with the actual type

hir_typeck_struct_expr_non_exhaustive =
    cannot create non-exhaustive {$what} using struct expression

hir_typeck_suggest_boxing_note = for more on the distinction between the stack and the heap, read https://doc.rust-lang.org/book/ch15-01-box.html, https://doc.rust-lang.org/rust-by-example/std/box.html, and https://doc.rust-lang.org/std/boxed/index.html

hir_typeck_suggest_boxing_when_appropriate = store this in the heap by calling `Box::new`

hir_typeck_suggest_ptr_null_mut = consider using `core::ptr::null_mut` instead

hir_typeck_trivial_cast = trivial {$numeric ->
    [true] numeric cast
    *[false] cast
    }: `{$expr_ty}` as `{$cast_ty}`
    .help = cast can be replaced by coercion; this might require a temporary variable

hir_typeck_union_pat_dotdot = `..` cannot be used in union patterns

hir_typeck_union_pat_multiple_fields = union patterns should have exactly one field

hir_typeck_use_is_empty =
    consider using the `is_empty` method on `{$expr_ty}` to determine if it contains anything

hir_typeck_yield_expr_outside_of_coroutine =
    yield expression outside of coroutine literal
