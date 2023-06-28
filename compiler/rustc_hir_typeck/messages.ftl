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

hir_typeck_const_select_must_be_const = this argument must be a `const fn`
    .help = consult the documentation on `const_eval_select` for more information

hir_typeck_const_select_must_be_fn = this argument must be a function item
    .note = expected a function item, found {$ty}
    .help = consult the documentation on `const_eval_select` for more information

hir_typeck_convert_to_str = try converting the passed type into a `&str`

hir_typeck_convert_using_method = try using `{$sugg}` to convert `{$found}` to `{$expected}`

hir_typeck_ctor_is_private = tuple struct constructor `{$def}` is private

hir_typeck_expected_default_return_type = expected `()` because of default return type

hir_typeck_expected_return_type = expected `{$expected}` because of return type

hir_typeck_field_multiply_specified_in_initializer =
    field `{$ident}` specified more than once
    .label = used more than once
    .previous_use_label = first use of `{$ident}`

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
hir_typeck_lang_start_expected_sig_note = the `start` lang item should have the signature `fn(fn() -> T, isize, *const *const u8, u8) -> isize`

hir_typeck_lang_start_incorrect_number_params = incorrect number of parameters for the `start` lang item
hir_typeck_lang_start_incorrect_number_params_note_expected_count = the `start` lang item should have four parameters, but found {$found_param_count}

hir_typeck_lang_start_incorrect_param = parameter {$param_num} of the `start` lang item is incorrect
    .suggestion = change the type from `{$found_ty}` to `{$expected_ty}`

hir_typeck_lang_start_incorrect_ret_ty = the return type of the `start` lang item is incorrect
    .suggestion = change the type from `{$found_ty}` to `{$expected_ty}`

hir_typeck_method_call_on_unknown_raw_pointee =
    cannot call a method on a raw pointer with an unknown pointee type

hir_typeck_missing_parentheses_in_range = can't call method `{$method_name}` on type `{$ty_str}`

hir_typeck_no_associated_item = no {$item_kind} named `{$item_name}` found for {$ty_prefix} `{$ty_str}`{$trait_missing_method ->
    [true] {""}
    *[other] {" "}in the current scope
}

hir_typeck_note_edition_guide = for more on editions, read https://doc.rust-lang.org/edition-guide

hir_typeck_op_trait_generic_params = `{$method_name}` must not have any generic parameters

hir_typeck_return_stmt_outside_of_fn_body =
    {$statement_kind} statement outside of function body
    .encl_body_label = the {$statement_kind} is part of this body...
    .encl_fn_label = ...not the enclosing function body

hir_typeck_struct_expr_non_exhaustive =
    cannot create non-exhaustive {$what} using struct expression

hir_typeck_suggest_boxing_note = for more on the distinction between the stack and the heap, read https://doc.rust-lang.org/book/ch15-01-box.html, https://doc.rust-lang.org/rust-by-example/std/box.html, and https://doc.rust-lang.org/std/boxed/index.html

hir_typeck_suggest_boxing_when_appropriate = store this in the heap by calling `Box::new`

hir_typeck_suggest_ptr_null_mut = consider using `core::ptr::null_mut` instead

hir_typeck_union_pat_dotdot = `..` cannot be used in union patterns

hir_typeck_union_pat_multiple_fields = union patterns should have exactly one field
hir_typeck_yield_expr_outside_of_generator =
    yield expression outside of generator literal
