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
    cannot move out of `{$desc}` beacause it is borrowed

borrowck_var_does_not_need_mut =
    variable does not need to be mutable
    .suggestion = remove this `mut`

borrowck_const_not_used_in_type_alias =
    const parameter `{$ct}` is part of concrete type but not used in parameter list for the `impl Trait` type alias

borrowck_var_cannot_escape_closure =
    captured variable cannot escape `FnMut` closure body
    .note = `FnMut` closures only have access to their captured variables while they are executing...
    .cannot_escape = ...therefore, they cannot allow references to captured variables to escape

borrowck_var_here_defined = variable defined here

borrowck_var_here_captured = variable captured here

borrowck_closure_inferred_mut =  inferred to be a `FnMut` closure

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

borrowck_data_moved_here =
    data moved here

borrowck_and_data_moved_here = ...and here

borrowck_moved_var_cannot_copy =
    move occurs because these variables have types that don't implement the `Copy` trait

borrowck_borrow_later_captured_by_trait_object =
    {$borrow_desc}borrow later captured here by trait object

borrowck_borrow_later_captured_by_closure =
    {$borrow_desc}borrow later captured here by closure

borrowck_borrow_later_used_by_call =
    {$borrow_desc}borrow later used by call

borrowck_borrow_later_stored_here =
    {$borrow_desc}borrow later stored here

borrowck_borrow_later_used_here =
    {$borrow_desc}borrow later used here

borrowck_used_here_by_closure =
    used here by closure

borrowck_trait_capture_borrow_in_later_iteration_loop =
    {$borrow_desc}borrow captured here by trait object, in later iteration of loop

borrowck_closure_capture_borrow_in_later_iteration_loop =
    {$borrow_desc}borrow captured here by closure, in later iteration of loop

borrowck_call_used_borrow_in_later_iteration_loop =
    {$borrow_desc}borrow used by call, in later iteration of loop

borrowck_used_borrow_in_later_iteration_loop =
    {$borrow_desc}borrow used here, in later iteration of loop

borrowck_bl_trait_capture_borrow_in_later_iteration_loop =
    {$borrow_desc}borrow later borrow captured here by trait object, in later iteration of loop

borrowck_bl_closure_capture_borrow_in_later_iteration_loop =
    {$borrow_desc}borrow later borrow captured here by closure, in later iteration of loop

borrowck_bl_call_used_borrow_in_later_iteration_loop =
    {$borrow_desc}borrow later borrow used by call, in later iteration of loop

borrowck_bl_borrow_later_stored_here =
    {$borrow_desc}borrow later borrow later stored here

borrowck_bl_used_borrow_in_later_iteration_loop =
    {$borrow_desc}borrow later borrow used here, in later iteration of loop

borrowck_drop_local_might_cause_borrow =
    {$borrow_desc}borrow might be used here, when `{$local_name}` is dropped and runs the {$dtor_desc} for {$type_desc}

borrowck_var_dropped_in_wrong_order =
    values in a scope are dropped in the opposite order they are defined

borrowck_temporary_access_to_borrow =
    a temporary with access to the {$borrow_desc}borrow is created here ...

borrowck_drop_temporary_might_cause_borrow_use = ... and the {$borrow_desc}borrow might be used here, when that temporary is dropped and runs the {$dtor_desc} for {$type_desc}

borrowck_consider_add_semicolon =
    consider adding semicolon after the expression so its temporaries are dropped sooner, before the local variables declared by the block are dropped

borrowck_consider_forcing_temporary_drop_sooner =
    the temporary is part of an expression at the end of a block;
    consider forcing this temporary to be dropped sooner, before the block's local variables are dropped

borrowck_perhaps_save_in_new_local_to_drop =
    for example, you could save the expression's value in a new local variable `x` and then make `x` be the expression at the end of the block

borrowck_outlive_constraint_need_borrow_for =
    {$category}requires that `{$desc}` is borrowed for `{$region_name}`

borrowck_outlive_constraint_need_borrow_lasts =
    {$category}requires that `{$borrow_desc}` lasts for `{$region_name}`
