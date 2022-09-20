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
    {$category ->
        [Assignment] assignment {""}
        [Return] returning this value {""}
        [Yield] yielding this value {""}
        [UseAsConst] using this value as a constant {""}
        [UseAsStatic] using this value as a static {""}
        [Cast] cast {""}
        [CallArgument] argument {""}
        [TypeAnnotation] type annotation {""}
        [ClosureBounds] closure body {""}
        [SizedBound] proving this value is `Sized` {""}
        [CopyBound] copying this value {""}
        [OpaqueType] opaque type {""}
        [ClosureUpvar] closure capture {""}
        [Usage] this usage {""}
        *[other] {""}
    }requires that `{$free_region_name}` must outlive `{$outlived_fr_name}`

borrowck_used_impl_require_static =
    the used `impl` has a `'static` requirement

borrowck_data_moved_here =
    data moved here

borrowck_and_data_moved_here = ...and here

borrowck_moved_var_cannot_copy =
    move occurs because these variables have types that don't implement the `Copy` trait

borrowck_used_here_by_closure =
    used here by closure

borrowck_drop_local_might_cause_borrow =
    {$borrow_desc ->
        [mutable] mutable {""}
        [immutable] immutable {""}
        [first] first {""}
        *[other] {""}
    }borrow might be used here, when `{$local_name}` is dropped and runs the {$dtor_desc} for {$type_desc}

borrowck_var_dropped_in_wrong_order =
    values in a scope are dropped in the opposite order they are defined

borrowck_temporary_access_to_borrow =
    a temporary with access to the {$borrow_desc ->
        [mutable] mutable {""}
        [immutable] immutable {""}
        [first] first {""}
        *[other] {""}
    }borrow is created here ...

borrowck_drop_temporary_might_cause_borrow_use = ... and the {$borrow_desc ->
        [mutable] mutable {""}
        [immutable] immutable {""}
        [first] first {""}
        *[other] {""}
    }borrow might be used here, when that temporary is dropped and runs the {$dtor_desc} for {$type_desc}

borrowck_consider_add_semicolon =
    consider adding semicolon after the expression so its temporaries are dropped sooner, before the local variables declared by the block are dropped

borrowck_consider_move_expression_end_of_block =
    for example, you could save the expression's value in a new local variable `x` and then make `x` be the expression at the end of the block

borrowck_consider_forcing_temporary_drop_sooner =
    the temporary is part of an expression at the end of a block;
    consider forcing this temporary to be dropped sooner, before the block's local variables are dropped

borrowck_perhaps_save_in_new_local_to_drop =
    for example, you could save the expression's value in a new local variable `x` and then make `x` be the expression at the end of the block

borrowck_outlive_constraint_need_borrow_for =
    {$category}requires that {$desc} is borrowed for `{$region_name}`

borrowck_outlive_constraint_need_borrow_lasts_for =
    {$category}requires that {$borrow_desc} is borrowed for `{$region_name}`

borrowck_consider_add_lifetime_bound =
    consider adding the following bound: `{$fr_name}: {$outlived_fr_name}`

borrowck_closure_cannot_invoke_again =
    closure cannot be invoked more than once because it moves the variable `{$place}` out of its environment

borrowck_closure_cannot_move_again =
    closure cannot be moved more than once as it is not `Copy` due to moving the variable `{$place}` out of its environment

borrowck_value_moved_here =
    value {$partially_str}moved{$move_msg ->
        [InClosure] {""} into closure
        *[other] {""}
    } here{$loop_message ->
        [InLoop] {""}, in previous iteration of loop
        *[other] {""}
    }

borrowck_consider_borrow_content_of_type =
    help: consider calling `.as_ref()` or `.as_mut()` to borrow the type's contents

borrowck_function_takes_self_ownership =
    this function takes ownership of the receiver `self`, which moves {$place_name}

borrowck_moved_by_method_call =
    {$place_name} {$partially_str}moved due to this method call{$loop_message ->
        [InLoop] {""}, in previous iteration of loop
        *[other] {""}
    }

borrowck_moved_by_implicit_call =
    {$place_name} {$partially_str}moved due to this implicit call to `.into_iter()`{$loop_message ->
        [InLoop] {""}, in previous iteration of loop
        *[other] {""}
    }

borrowck_lhs_moved_by_operator_call =
    calling this operator moves the left-hand side

borrowck_moved_by_operator_use =
    {$place_name} {$partially_str}moved due to usage in operator{$loop_message ->
        [InLoop] {""}, in previous iteration of loop
        *[other] {""}
    }

borrowck_moved_fnonce_value =
    this value implements `FnOnce`, which causes it to be moved when called

borrowck_moved_by_call =
    {$place_name} {$partially_str}moved due to this call{$loop_message ->
        [InLoop] {""}, in previous iteration of loop
        *[other] {""}
    }

borrowck_type_not_impl_Copy =
    {$move_prefix}move occurs because {$place_desc ->
        [value] value
        *[other] {$place_desc}
    } has type `{$ty}`, which does not implement the `Copy` trait

borrowck_outlive_constraint_need_borrow_lasts =
    {$category}requires that `{$borrow_desc}` lasts for `{$region_name}`

borrowck_require_mutable_binding =
    calling `{$place}` requires mutable binding due to {$reason ->
        [borrow] mutable borrow of `{$upvar}`
        *[mutation] possible mutation of `{$upvar}`
    }

borrowck_cannot_act =
    cannot {$act}

borrowck_expects_fnmut_not_fn =
    change this to accept `FnMut` instead of `Fn`

borrowck_expects_fn_not_fnmut =
    expects `Fn` instead of `FnMut`

borrowck_empty_label = {""}

borrowck_in_this_closure =
    in this closure

borrowck_return_fnmut =
    change this to return `FnMut` instead of `Fn`

borrowck_name_this_region =
    let's call this `{$rg_name}`

borrowck_lifetime_appears_in_type =
    lifetime `{$rg_name}` appears in the type {$type_name}

borrowck_return_type_has_lifetime =
    return type{$mir_description ->
        [Block] {""} of async block
        [Closure] {""} of async closure
        [Fn] {""} of async function
        [Gen] {""} of generator
        [None] {""} of closure
        *[other] {""}
    } `{$type_name}` contains a lifetime `{$rg_name}`

borrowck_lifetime_appears_in_type_of =
    lifetime `{$rg_name}` appears in the type of `{$upvar_name}`

borrowck_return_type_is_type =
    return type{$mir_description ->
        [Block] {""} of async block
        [Closure] {""} of async closure
        [Fn] {""} of async function
        [Gen] {""} of generator
        [None] {""} of closure
        *[other] {""}
    } is {$type_name}

borrowck_yield_type_is_type =
    yield type is {$type_name}

borrowck_lifetime_appears_here_in_impl =
    lifetime `{$rg_name}` appears in the `impl`'s {$location}

borrowck_type_parameter_not_used_in_trait_type_alias =
    type parameter `{$ty}` is part of concrete type but not used in parameter list for the `impl Trait` type alias

borrowck_non_defining_opaque_type =
    non-defining opaque type use in defining scope

borrowck_lifetime_not_used_in_trait_type_alias =
    lifetime `{$r}` is part of concrete type but not used in parameter list of the `impl Trait` type alias

borrowck_used_non_generic_for_generic =
    used non-generic {$descr ->
    [lifetime] lifetime
    [type] type
    *[constant] constant
    } `{$arg}` for generic parameter

borrowck_cannot_use_static_lifetime_here =
    cannot use static lifetime; use a bound lifetime instead or remove the lifetime parameter from the opaque type

borrowck_define_inline_constant_type =
    defining inline constant type: {$type_name}

borrowck_define_const_type =
    defining constant type: {$type_name}

borrowck_define_type =
    defining type: {$type_name}

borrowck_define_type_with_generator_substs =
    defining type: {$type_name} with generator substs {$subsets}

borrowck_define_type_with_closure_substs =
    defining type: {$type_name} with closure substs {$subsets}

borrowck_borrowed_temporary_value_dropped =
    temporary value dropped while borrowed

borrowck_thread_local_outlive_function =
    thread-local variable borrowed past end of function

borrowck_closure_borrowing_outlive_function =
    {$closure_kind} may outlive the current function, but it borrows {$borrowed_path}, which is owned by the current function
    .label = may outlive borrowed value {$borrowed_path}
    .path_label = {$borrowed_path} is borrowed here

borrowck_cannot_return_ref_to_local =
    cannot {$return_kind} {$reference} {$local}
    .label = {$return_kind}s a {$reference} data owned by the current function

borrowck_path_does_not_live_long_enough =
    {$path} does not live long enough

borrowck_cannot_borrow_across_destructor =
    borrow may still be in use when destructor runs

borrowck_cannot_borrow_across_generator_yield =
    borrow may still be in use when generator yields
    .label = possible yield occurs here

borrowck_cannot_move_out_of_interior_of_drop =
    cannot move out of type `{$container_ty}`, which implements the `Drop` trait
    .label = cannot move out of here

borrowck_cannot_assign_to_borrowed =
    cannot assign to {$desc ->
        [value] value
        *[other] {$desc}
    } because it is borrowed
    .label = assignment to borrowed {$desc ->
        [value] value
        *[other] {$desc}
    } occurs here
    .borrow_here_label = borrow of {$desc ->
        [value] value
        *[other] {$desc}
    } occurs here

borrowck_cannot_uniquely_borrow_by_two_closures =
    two closures require unique access to {$desc ->
        [value] value
        *[other] {$desc}
    } at the same time
    .label = borrow from first closure ends here
    .new_span_label = second closure is constructed here

borrowck_first_closure_constructed_here =
    first closure is constructed here

borrowck_closures_constructed_here =
    closures are constructed here in different iterations of loop

borrowck_cannot_use_when_mutably_borrowed =
    cannot use {$desc ->
        [value] value
        *[other] {$desc}
    } because it was mutably borrowed
    .label = use of borrowed {$borrow_desc ->
        [value] value
        *[other] {$borrow_desc}
    }
    .borrow_span_label = borrow of {$borrow_desc ->
        [value] value
        *[other] {$borrow_desc}
    } occurs here

borrowck_cannot_move_when_borrowed =
    cannot move out of {$desc ->
        [value] value
        *[other] {$desc}
    } because it is borrowed
