infer_opaque_hidden_type =
    opaque type's hidden type cannot be another opaque type from the same scope
    .label = one of the two opaque types used here has to be outside its defining scope
    .opaque_type = opaque type whose hidden type is being assigned
    .hidden_type = opaque type being used as hidden type

infer_type_annotations_needed = {$source_kind ->
    [closure] type annotations needed for the closure `{$source_name}`
    [normal] type annotations needed for `{$source_name}`
    *[other] type annotations needed
}
    .label = type must be known at this point

infer_label_bad = {$bad_kind ->
    *[other] cannot infer type
    [more_info] cannot infer {$prefix_kind ->
        *[type] type for {$prefix}
        [const_with_param] the value of const parameter
        [const] the value of the constant
    } `{$name}`{$has_parent ->
        [true] {" "}declared on the {$parent_prefix} `{$parent_name}`
        *[false] {""}
    }
}

infer_source_kind_subdiag_let = {$kind ->
    [with_pattern] consider giving `{$name}` an explicit type
    [closure] consider giving this closure parameter an explicit type
    *[other] consider giving this pattern a type
}{$x_kind ->
    [has_name] , where the {$prefix_kind ->
        *[type] type for {$prefix}
        [const_with_param] the value of const parameter
        [const] the value of the constant
    } `{$arg_name}` is specified
    [underscore] , where the placeholders `_` are specified
    *[empty] {""}
}

infer_source_kind_subdiag_generic_label =
    cannot infer {$is_type ->
    [true] type
    *[false] the value
    } of the {$is_type ->
    [true] type
    *[false] const
    } {$parent_exists ->
    [true] parameter `{$param_name}` declared on the {$parent_prefix} `{$parent_name}`
    *[false] parameter {$param_name}
    }

infer_source_kind_subdiag_generic_suggestion =
    consider specifying the generic {$arg_count ->
    [one] argument
    *[other] arguments
    }

infer_source_kind_fully_qualified =
    try using a fully qualified path to specify the expected types

infer_source_kind_closure_return =
    try giving this closure an explicit return type

# generator_kind  may need to be translated
infer_need_type_info_in_generator =
    type inside {$generator_kind ->
    [async_block] `async` block
    [async_closure] `async` closure
    [async_fn] `async fn` body
    *[generator] generator
    } must be known in this context


infer_subtype = ...so that the {$requirement ->
    [method_compat] method type is compatible with trait
    [type_compat] associated type is compatible with trait
    [const_compat] const is compatible with trait
    [expr_assignable] expression is assignable
    [if_else_different] `if` and `else` have incompatible types
    [no_else] `if` missing an `else` returns `()`
    [fn_main_correct_type] `main` function has the correct type
    [fn_start_correct_type] `#[start]` function has the correct type
    [intristic_correct_type] intrinsic has the correct type
    [method_correct_type] method receiver has the correct type
    *[other] types are compatible
}
infer_subtype_2 = ...so that {$requirement ->
    [method_compat] method type is compatible with trait
    [type_compat] associated type is compatible with trait
    [const_compat] const is compatible with trait
    [expr_assignable] expression is assignable
    [if_else_different] `if` and `else` have incompatible types
    [no_else] `if` missing an `else` returns `()`
    [fn_main_correct_type] `main` function has the correct type
    [fn_start_correct_type] `#[start]` function has the correct type
    [intristic_correct_type] intrinsic has the correct type
    [method_correct_type] method receiver has the correct type
    *[other] types are compatible
}

infer_reborrow = ...so that reference does not outlive borrowed content
infer_reborrow_upvar = ...so that closure can access `{$name}`
infer_relate_object_bound = ...so that it can be closed over into an object
infer_reference_outlives_referent = ...so that the reference type `{$name}` does not outlive the data it points at
infer_relate_param_bound = ...so that the type `{$name}` will meet its required lifetime bounds{$continues ->
    [true] ...
    *[false] {""}
}
infer_relate_param_bound_2 = ...that is required by this bound
infer_relate_region_param_bound = ...so that the declared lifetime parameter bounds are satisfied
infer_compare_impl_item_obligation = ...so that the definition in impl matches the definition from the trait
infer_ascribe_user_type_prove_predicate = ...so that the where clause holds

infer_nothing = {""}

infer_lifetime_mismatch = lifetime mismatch

infer_declared_different = this parameter and the return type are declared with different lifetimes...
infer_data_returned = ...but data{$label_var1_exists ->
    [true] {" "}from `{$label_var1}`
    *[false] {""}
} is returned here

infer_data_lifetime_flow = ...but data with one lifetime flows into the other here
infer_declared_multiple = this type is declared with multiple lifetimes...
infer_types_declared_different = these two types are declared with different lifetimes...
infer_data_flows = ...but data{$label_var1_exists ->
    [true] {" "}from `{$label_var1}`
    *[false] -> {""}
} flows{$label_var2_exists ->
    [true] {" "}into `{$label_var2}`
    *[false] -> {""}
} here

infer_lifetime_param_suggestion = consider introducing a named lifetime parameter{$is_impl ->
    [true] {" "}and update trait if needed
    *[false] {""}
}
infer_lifetime_param_suggestion_elided = each elided lifetime in input position becomes a distinct lifetime

infer_region_explanation = {$pref_kind ->
    *[should_not_happen] [{$pref_kind}]
    [ref_valid_for] ...the reference is valid for
    [content_valid_for] ...but the borrowed content is only valid for
    [type_obj_valid_for] object type is valid for
    [source_pointer_valid_for] source pointer is only valid for
    [type_satisfy] type must satisfy
    [type_outlive] type must outlive
    [lf_param_instantiated_with] lifetime parameter instantiated with
    [lf_param_must_outlive] but lifetime parameter must outlive
    [lf_instantiated_with] lifetime instantiated with
    [lf_must_outlive] but lifetime must outlive
    [pointer_valid_for] the pointer is valid for
    [data_valid_for] but the referenced data is only valid for
    [empty] {""}
}{$pref_kind ->
    [empty] {""}
    *[other] {" "}
}{$desc_kind ->
    *[should_not_happen] [{$desc_kind}]
    [restatic] the static lifetime
    [revar] lifetime {$desc_arg}
    [as_defined] the lifetime `{$desc_arg}` as defined here
    [as_defined_anon] the anonymous lifetime as defined here
    [defined_here] the anonymous lifetime defined here
    [anon_num_here] the anonymous lifetime #{$desc_num_arg} defined here
    [defined_here_reg] the lifetime `{$desc_arg}` as defined here
}{$suff_kind ->
    *[should_not_happen] [{$suff_kind}]
    [empty]{""}
    [continues] ...
    [req_by_binding] {" "}as required by this binding
}

infer_outlives_content = lifetime of reference outlives lifetime of borrowed content...
infer_outlives_bound = lifetime of the source pointer does not outlive lifetime bound of the object type
infer_fullfill_req_lifetime = the type `{$ty}` does not fulfill the required lifetime
infer_lf_bound_not_satisfied = lifetime bound not satisfied
infer_borrowed_too_long = a value of type `{$ty}` is borrowed for too long
infer_ref_longer_than_data = in type `{$ty}`, reference has a longer lifetime than the data it references

infer_mismatched_static_lifetime = incompatible lifetime on type
infer_does_not_outlive_static_from_impl = ...does not necessarily outlive the static lifetime introduced by the compatible `impl`
infer_implicit_static_lifetime_note = this has an implicit `'static` lifetime requirement
infer_implicit_static_lifetime_suggestion = consider relaxing the implicit `'static` requirement
infer_msl_introduces_static = introduces a `'static` lifetime requirement
infer_msl_unmet_req = because this has an unmet lifetime requirement
infer_msl_trait_note = this has an implicit `'static` lifetime requirement
infer_msl_trait_sugg = consider relaxing the implicit `'static` requirement
infer_suggest_add_let_for_letchains = consider adding `let`

infer_explicit_lifetime_required_with_ident = explicit lifetime required in the type of `{$simple_ident}`
    .label = lifetime `{$named}` required

infer_explicit_lifetime_required_with_param_type = explicit lifetime required in parameter type
    .label = lifetime `{$named}` required

infer_explicit_lifetime_required_sugg_with_ident = add explicit lifetime `{$named}` to the type of `{$simple_ident}`

infer_explicit_lifetime_required_sugg_with_param_type = add explicit lifetime `{$named}` to type

infer_actual_impl_expl_expected_signature_two = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}closure with signature `{$ty_or_sig}` must implement `{$trait_path}`, for any two lifetimes `'{$lifetime_1}` and `'{$lifetime_2}`...
infer_actual_impl_expl_expected_signature_any = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}closure with signature `{$ty_or_sig}` must implement `{$trait_path}`, for any lifetime `'{$lifetime_1}`...
infer_actual_impl_expl_expected_signature_some = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}closure with signature `{$ty_or_sig}` must implement `{$trait_path}`, for some specific lifetime `'{$lifetime_1}`...
infer_actual_impl_expl_expected_signature_nothing = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}closure with signature `{$ty_or_sig}` must implement `{$trait_path}`
infer_actual_impl_expl_expected_passive_two = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$trait_path}` would have to be implemented for the type `{$ty_or_sig}`, for any two lifetimes `'{$lifetime_1}` and `'{$lifetime_2}`...
infer_actual_impl_expl_expected_passive_any = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$trait_path}` would have to be implemented for the type `{$ty_or_sig}`, for any lifetime `'{$lifetime_1}`...
infer_actual_impl_expl_expected_passive_some = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$trait_path}` would have to be implemented for the type `{$ty_or_sig}`, for some specific lifetime `'{$lifetime_1}`...
infer_actual_impl_expl_expected_passive_nothing = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$trait_path}` would have to be implemented for the type `{$ty_or_sig}`
infer_actual_impl_expl_expected_other_two = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$ty_or_sig}` must implement `{$trait_path}`, for any two lifetimes `'{$lifetime_1}` and `'{$lifetime_2}`...
infer_actual_impl_expl_expected_other_any = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$ty_or_sig}` must implement `{$trait_path}`, for any lifetime `'{$lifetime_1}`...
infer_actual_impl_expl_expected_other_some = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$ty_or_sig}` must implement `{$trait_path}`, for some specific lifetime `'{$lifetime_1}`...
infer_actual_impl_expl_expected_other_nothing = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$ty_or_sig}` must implement `{$trait_path}`

infer_actual_impl_expl_but_actually_implements_trait = ...but it actually implements `{$trait_path}`{$has_lifetime ->
    [true] , for some specific lifetime `'{$lifetime}`
    *[false] {""}
}
infer_actual_impl_expl_but_actually_implemented_for_ty = ...but `{$trait_path}` is actually implemented for the type `{$ty}`{$has_lifetime ->
    [true] , for some specific lifetime `'{$lifetime}`
    *[false] {""}
}
infer_actual_impl_expl_but_actually_ty_implements = ...but `{$ty}` actually implements `{$trait_path}`{$has_lifetime ->
    [true] , for some specific lifetime `'{$lifetime}`
    *[false] {""}
}

infer_trait_placeholder_mismatch = implementation of `{$trait_def_id}` is not general enough
    .label_satisfy = doesn't satisfy where-clause
    .label_where = due to a where-clause on `{$def_id}`...
    .label_dup = implementation of `{$trait_def_id}` is not general enough

infer_trait_impl_diff = `impl` item signature doesn't match `trait` item signature
    .found = found `{$found}`
    .expected = expected `{$expected}`
    .expected_found = expected signature `{$expected}`
               {"   "}found signature `{$found}`

infer_tid_rel_help = verify the lifetime relationships in the `trait` and `impl` between the `self` argument, the other inputs and its output
infer_tid_consider_borrowing = consider borrowing this type parameter in the trait
infer_tid_param_help = the lifetime requirements from the `impl` do not correspond to the requirements in the `trait`

infer_dtcs_has_lifetime_req_label = this has an implicit `'static` lifetime requirement
infer_dtcs_introduces_requirement = calling this method introduces the `impl`'s `'static` requirement
infer_dtcs_has_req_note = the used `impl` has a `'static` requirement
infer_dtcs_suggestion = consider relaxing the implicit `'static` requirement

infer_but_calling_introduces = {$has_param_name ->
    [true] `{$param_name}`
    *[false] `fn` parameter
} has {$lifetime_kind ->
    [true] lifetime `{$lifetime}`
    *[false] an anonymous lifetime `'_`
} but calling `{$assoc_item}` introduces an implicit `'static` lifetime requirement
    .label1 = {$has_lifetime ->
        [true] lifetime `{$lifetime}`
        *[false] an anonymous lifetime `'_`
    }
    .label2 = ...is used and required to live as long as `'static` here because of an implicit lifetime bound on the {$has_impl_path ->
        [true] `impl` of `{$impl_path}`
        *[false] inherent `impl`
    }

infer_but_needs_to_satisfy = {$has_param_name ->
    [true] `{$param_name}`
    *[false] `fn` parameter
} has {$has_lifetime ->
    [true] lifetime `{$lifetime}`
    *[false] an anonymous lifetime `'_`
} but it needs to satisfy a `'static` lifetime requirement
    .influencer = this data with {$has_lifetime ->
        [true] lifetime `{$lifetime}`
        *[false] an anonymous lifetime `'_`
    }...
    .require = {$spans_empty ->
        *[true] ...is used and required to live as long as `'static` here
        [false] ...and is required to live as long as `'static` here
    }
    .used_here = ...is used here...
    .introduced_by_bound = `'static` lifetime requirement introduced by this bound

infer_more_targeted = {$has_param_name ->
    [true] `{$param_name}`
    *[false] `fn` parameter
} has {$has_lifetime ->
    [true] lifetime `{$lifetime}`
    *[false] an anonymous lifetime `'_`
} but calling `{$ident}` introduces an implicit `'static` lifetime requirement

infer_ril_introduced_here = `'static` requirement introduced here
infer_ril_introduced_by = requirement introduced by this return type
infer_ril_because_of = because of this returned expression
infer_ril_static_introduced_by = "`'static` lifetime requirement introduced by the return type

infer_where_remove = remove the `where` clause
infer_where_copy_predicates = copy the `where` clause predicates from the trait

infer_srs_remove_and_box = consider removing this semicolon and boxing the expressions
infer_srs_remove = consider removing this semicolon
infer_srs_add = consider returning the local binding `{$ident}`
infer_srs_add_one = consider returning one of these bindings

infer_await_both_futures = consider `await`ing on both `Future`s
infer_await_future = consider `await`ing on the `Future`
infer_await_note = calling an async function returns a future

infer_prlf_defined_with_sub = the lifetime `{$sub_symbol}` defined here...
infer_prlf_defined_without_sub = the lifetime defined here...
infer_prlf_must_oultive_with_sup = ...must outlive the lifetime `{$sup_symbol}` defined here
infer_prlf_must_oultive_without_sup = ...must outlive the lifetime defined here
infer_prlf_known_limitation = this is a known limitation that will be removed in the future (see issue #100013 <https://github.com/rust-lang/rust/issues/100013> for more information)
