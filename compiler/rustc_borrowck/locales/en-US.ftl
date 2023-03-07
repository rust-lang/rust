borrowck_move_unsized =
    cannot move a value of type `{$ty}`
    .label = the size of `{$ty}` cannot be statically determined

borrowck_higher_ranked_lifetime_error =
    higher-ranked lifetime error

borrowck_could_not_prove =
    could not prove `{$predicate}`

borrowck_could_not_normalize =
    could not normalize `{$value}`

borrowck_higher_ranked_subtype_error =
    higher-ranked subtype error

borrowck_generic_does_not_live_long_enough =
    `{$kind}` does not live long enough

borrowck_move_borrowed =
    cannot move out of `{$desc}` because it is borrowed

borrowck_var_does_not_need_mut =
    variable does not need to be mutable
    .suggestion = remove this `mut`

borrowck_var_cannot_escape_closure =
    captured variable cannot escape `FnMut` closure body
    .note = `FnMut` closures only have access to their captured variables while they are executing...
    .cannot_escape = ...therefore, they cannot allow references to captured variables to escape

borrowck_var_here_defined = variable defined here

borrowck_var_here_captured = variable captured here

borrowck_closure_inferred_mut = inferred to be a `FnMut` closure

borrowck_returned_closure_escaped =
    returns a closure that contains a reference to a captured variable, which then escapes the closure body

borrowck_returned_async_block_escaped =
    returns an `async` block that contains a reference to a captured variable, which then escapes the closure body

borrowck_returned_ref_escaped =
    returns a reference to a captured variable which escapes the closure body

borrowck_lifetime_constraints_error =
    lifetime may not live long enough

borrowck_returned_lifetime_wrong =
    {$mir_def_name} was supposed to return data with lifetime `{$outlived_fr_name}` but it is returning data with lifetime `{$fr_name}`

borrowck_returned_lifetime_short =
    {$category_desc}requires that `{$free_region_name}` must outlive `{$outlived_fr_name}`

borrowck_used_impl_require_static =
    the used `impl` has a `'static` requirement

borrowck_capture_kind_label =
    capture is {$kind_desc} because of use here

borrowck_var_borrow_by_use_place_in_generator =
    borrow occurs due to use of {$place} in closure in generator

borrowck_var_borrow_by_use_place_in_closure =
    borrow occurs due to use of {$place} in closure

borrowck_var_borrow_by_use_place =
    borrow occurs due to use of {$place}

borrowck_borrow_due_to_use_generator =
    borrow occurs due to use in generator

borrowck_use_due_to_use_generator =
    use occurs due to use in generator

borrowck_assign_due_to_use_generator =
    assign occurs due to use in generator

borrowck_assign_part_due_to_use_generator =
    assign to part occurs due to use in generator

borrowck_borrow_due_to_use_closure =
    borrow occurs due to use in closure

borrowck_use_due_to_use_closure =
    use occurs due to use in closure

borrowck_assign_due_to_use_closure =
    assignment occurs due to use in closure

borrowck_assign_part_due_to_use_closure =
    assignment to part occurs due to use in closure

borrowck_capture_immute =
    capture is immutable because of use here

borrowck_capture_mut =
    capture is mutable because of use here

borrowck_capture_move =
    capture is moved because of use here

borrowck_var_move_by_use_place_in_generator =
    move occurs due to use of {$place} in generator

borrowck_var_move_by_use_place_in_closure =
    move occurs due to use of {$place} in closure

borrowck_cannot_move_when_borrowed =
    cannot move out of {$place ->
        [value] value
        *[other] {$place}
    } because it is borrowed
    .label = borrow of {$borrow_place ->
        [value] value
        *[other] {$borrow_place}
    } occurs here
    .move_label = move out of {$value_place ->
        [value] value
        *[other] {$value_place}
    } occurs here

borrowck_opaque_type_non_generic_param =
    expected generic {$kind} parameter, found `{$ty}`
    .label = {STREQ($ty, "'static") ->
        [true] cannot use static lifetime; use a bound lifetime instead or remove the lifetime parameter from the opaque type
        *[other] this generic parameter must be used with a generic {$kind} parameter
    }
