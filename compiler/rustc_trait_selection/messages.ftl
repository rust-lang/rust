trait_selection_actual_impl_expl_but_actually_implemented_for_ty = ...but `{$trait_path}` is actually implemented for the type `{$ty}`{$has_lifetime ->
    [true] , for some specific lifetime `'{$lifetime}`
    *[false] {""}
}
trait_selection_actual_impl_expl_but_actually_implements_trait = ...but it actually implements `{$trait_path}`{$has_lifetime ->
    [true] , for some specific lifetime `'{$lifetime}`
    *[false] {""}
}
trait_selection_actual_impl_expl_but_actually_ty_implements = ...but `{$ty}` actually implements `{$trait_path}`{$has_lifetime ->
    [true] , for some specific lifetime `'{$lifetime}`
    *[false] {""}
}

trait_selection_actual_impl_expl_expected_other_any = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$ty_or_sig}` must implement `{$trait_path}`, for any lifetime `'{$lifetime_1}`...
trait_selection_actual_impl_expl_expected_other_nothing = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$ty_or_sig}` must implement `{$trait_path}`

trait_selection_actual_impl_expl_expected_other_some = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$ty_or_sig}` must implement `{$trait_path}`, for some specific lifetime `'{$lifetime_1}`...
trait_selection_actual_impl_expl_expected_other_two = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$ty_or_sig}` must implement `{$trait_path}`, for any two lifetimes `'{$lifetime_1}` and `'{$lifetime_2}`...
trait_selection_actual_impl_expl_expected_passive_any = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$trait_path}` would have to be implemented for the type `{$ty_or_sig}`, for any lifetime `'{$lifetime_1}`...
trait_selection_actual_impl_expl_expected_passive_nothing = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$trait_path}` would have to be implemented for the type `{$ty_or_sig}`
trait_selection_actual_impl_expl_expected_passive_some = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$trait_path}` would have to be implemented for the type `{$ty_or_sig}`, for some specific lifetime `'{$lifetime_1}`...
trait_selection_actual_impl_expl_expected_passive_two = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}`{$trait_path}` would have to be implemented for the type `{$ty_or_sig}`, for any two lifetimes `'{$lifetime_1}` and `'{$lifetime_2}`...
trait_selection_actual_impl_expl_expected_signature_any = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}closure with signature `{$ty_or_sig}` must implement `{$trait_path}`, for any lifetime `'{$lifetime_1}`...
trait_selection_actual_impl_expl_expected_signature_nothing = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}closure with signature `{$ty_or_sig}` must implement `{$trait_path}`
trait_selection_actual_impl_expl_expected_signature_some = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}closure with signature `{$ty_or_sig}` must implement `{$trait_path}`, for some specific lifetime `'{$lifetime_1}`...
trait_selection_actual_impl_expl_expected_signature_two = {$leading_ellipsis ->
    [true] ...
    *[false] {""}
}closure with signature `{$ty_or_sig}` must implement `{$trait_path}`, for any two lifetimes `'{$lifetime_1}` and `'{$lifetime_2}`...
trait_selection_adjust_signature_borrow = consider adjusting the signature so it borrows its {$len ->
        [one] argument
        *[other] arguments
    }

trait_selection_adjust_signature_remove_borrow = consider adjusting the signature so it does not borrow its {$len ->
        [one] argument
        *[other] arguments
    }

trait_selection_ascribe_user_type_prove_predicate = ...so that the where clause holds

trait_selection_await_both_futures = consider `await`ing on both `Future`s
trait_selection_await_future = consider `await`ing on the `Future`
trait_selection_await_note = calling an async function returns a future

trait_selection_but_calling_introduces = {$has_param_name ->
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

trait_selection_but_needs_to_satisfy = {$has_param_name ->
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

trait_selection_closure_fn_mut_label = closure is `{$trait_prefix}FnMut` because it mutates the variable `{$place}` here

trait_selection_closure_fn_once_label = closure is `{$trait_prefix}FnOnce` because it moves the variable `{$place}` out of its environment

trait_selection_closure_kind_mismatch = expected a closure that implements the `{$trait_prefix}{$expected}` trait, but this closure only implements `{$trait_prefix}{$found}`
    .label = this closure implements `{$trait_prefix}{$found}`, not `{$trait_prefix}{$expected}`

trait_selection_closure_kind_requirement = the requirement to implement `{$trait_prefix}{$expected}` derives from here

trait_selection_compare_impl_item_obligation = ...so that the definition in impl matches the definition from the trait
trait_selection_consider_specifying_length = consider specifying the actual array length
trait_selection_coro_closure_not_fn = {$coro_kind}closure does not implement `{$kind}` because it captures state from its environment

trait_selection_data_flows = ...but data{$label_var1_exists ->
    [true] {" "}from `{$label_var1}`
    *[false] {""}
} flows{$label_var2_exists ->
    [true] {" "}into `{$label_var2}`
    *[false] {""}
} here

trait_selection_data_lifetime_flow = ...but data with one lifetime flows into the other here
trait_selection_data_returned = ...but data{$label_var1_exists ->
    [true] {" "}from `{$label_var1}`
    *[false] {""}
} is returned here

trait_selection_declared_different = this parameter and the return type are declared with different lifetimes...
trait_selection_declared_multiple = this type is declared with multiple lifetimes...
trait_selection_disallowed_positional_argument = positional format arguments are not allowed here
    .help = only named format arguments with the name of one of the generic types are allowed in this context

trait_selection_does_not_outlive_static_from_impl = ...does not necessarily outlive the static lifetime introduced by the compatible `impl`
trait_selection_dtcs_has_lifetime_req_label = this has an implicit `'static` lifetime requirement
trait_selection_dtcs_has_req_note = the used `impl` has a `'static` requirement
trait_selection_dtcs_introduces_requirement = calling this method introduces the `impl`'s `'static` requirement
trait_selection_dtcs_suggestion = consider relaxing the implicit `'static` requirement

trait_selection_explicit_lifetime_required_sugg_with_ident = add explicit lifetime `{$named}` to the type of `{$simple_ident}`

trait_selection_explicit_lifetime_required_sugg_with_param_type = add explicit lifetime `{$named}` to type

trait_selection_explicit_lifetime_required_with_ident = explicit lifetime required in the type of `{$simple_ident}`
    .label = lifetime `{$named}` required

trait_selection_explicit_lifetime_required_with_param_type = explicit lifetime required in parameter type
    .label = lifetime `{$named}` required

trait_selection_fn_consider_casting = consider casting the fn item to a fn pointer: `{$casting}`

trait_selection_fn_consider_casting_both = consider casting both fn items to fn pointers using `as {$sig}`

trait_selection_fn_uniq_types = different fn items have unique types, even if their signatures are the same
trait_selection_fps_cast = consider casting to a fn pointer
trait_selection_fps_cast_both = consider casting both fn items to fn pointers using `as {$expected_sig}`

trait_selection_fps_items_are_distinct = fn items are distinct from fn pointers
trait_selection_fps_remove_ref = consider removing the reference
trait_selection_fps_use_ref = consider using a reference
trait_selection_fulfill_req_lifetime = the type `{$ty}` does not fulfill the required lifetime

trait_selection_ignored_diagnostic_option = `{$option_name}` is ignored due to previous definition of `{$option_name}`
    .other_label = `{$option_name}` is first declared here
    .label = `{$option_name}` is already declared here

trait_selection_implicit_static_lifetime_note = this has an implicit `'static` lifetime requirement
trait_selection_implicit_static_lifetime_suggestion = consider relaxing the implicit `'static` requirement
trait_selection_inherent_projection_normalization_overflow = overflow evaluating associated type `{$ty}`

trait_selection_invalid_format_specifier = invalid format specifier
    .help = no format specifier are supported in this position

trait_selection_label_bad = {$bad_kind ->
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

trait_selection_lf_bound_not_satisfied = lifetime bound not satisfied
trait_selection_lifetime_mismatch = lifetime mismatch

trait_selection_lifetime_param_suggestion = consider {$is_reuse ->
    [true] reusing
    *[false] introducing
} a named lifetime parameter{$is_impl ->
    [true] {" "}and update trait if needed
    *[false] {""}
}
trait_selection_lifetime_param_suggestion_elided = each elided lifetime in input position becomes a distinct lifetime

trait_selection_malformed_on_unimplemented_attr = malformed `on_unimplemented` attribute
    .help = only `message`, `note` and `label` are allowed as options
    .label = invalid option found here

trait_selection_meant_byte_literal = if you meant to write a byte literal, prefix with `b`
trait_selection_meant_char_literal = if you meant to write a `char` literal, use single quotes
trait_selection_meant_str_literal = if you meant to write a string literal, use double quotes
trait_selection_mismatched_static_lifetime = incompatible lifetime on type
trait_selection_missing_options_for_on_unimplemented_attr = missing options for `on_unimplemented` attribute
    .help = at least one of the `message`, `note` and `label` options are expected

trait_selection_msl_introduces_static = introduces a `'static` lifetime requirement
trait_selection_msl_unmet_req = because this has an unmet lifetime requirement

trait_selection_negative_positive_conflict = found both positive and negative implementation of trait `{$trait_desc}`{$self_desc ->
        [none] {""}
       *[default] {" "}for type `{$self_desc}`
    }:
    .negative_implementation_here = negative implementation here
    .negative_implementation_in_crate = negative implementation in crate `{$negative_impl_cname}`
    .positive_implementation_here = positive implementation here
    .positive_implementation_in_crate = positive implementation in crate `{$positive_impl_cname}`

trait_selection_nothing = {""}

trait_selection_oc_cant_coerce_force_inline =
    cannot coerce functions which must be inlined to function pointers
trait_selection_oc_cant_coerce_intrinsic = cannot coerce intrinsics to function pointers
trait_selection_oc_closure_selfref = closure/coroutine type that references itself
trait_selection_oc_const_compat = const not compatible with trait
trait_selection_oc_fn_lang_correct_type = {$lang_item_name ->
        [panic_impl] `#[panic_handler]`
        *[lang_item_name] lang item `{$lang_item_name}`
    } function has wrong type
trait_selection_oc_fn_main_correct_type = `main` function has wrong type
trait_selection_oc_generic = mismatched types

trait_selection_oc_if_else_different = `if` and `else` have incompatible types
trait_selection_oc_intrinsic_correct_type = intrinsic has wrong type
trait_selection_oc_match_compat = `match` arms have incompatible types
trait_selection_oc_method_compat = method not compatible with trait
trait_selection_oc_method_correct_type = mismatched `self` parameter type
trait_selection_oc_no_diverge = `else` clause of `let...else` does not diverge
trait_selection_oc_no_else = `if` may be missing an `else` clause
trait_selection_oc_try_compat = `?` operator has incompatible types
trait_selection_oc_type_compat = type not compatible with trait

trait_selection_opaque_captures_lifetime = hidden type for `{$opaque_ty}` captures lifetime that does not appear in bounds
    .label = opaque type defined here
trait_selection_opaque_type_non_generic_param =
    expected generic {$kind} parameter, found `{$arg}`
    .label = {STREQ($arg, "'static") ->
        [true] cannot use static lifetime; use a bound lifetime instead or remove the lifetime parameter from the opaque type
        *[other] this generic parameter must be used with a generic {$kind} parameter
    }

trait_selection_outlives_bound = lifetime of the source pointer does not outlive lifetime bound of the object type
trait_selection_outlives_content = lifetime of reference outlives lifetime of borrowed content...

trait_selection_precise_capturing_existing = add `{$new_lifetime}` to the `use<...>` bound to explicitly capture it
trait_selection_precise_capturing_new = add a `use<...>` bound to explicitly capture `{$new_lifetime}`

trait_selection_precise_capturing_new_but_apit = add a `use<...>` bound to explicitly capture `{$new_lifetime}` after turning all argument-position `impl Trait` into type parameters, noting that this possibly affects the API of this crate

trait_selection_precise_capturing_overcaptures = use the precise capturing `use<...>` syntax to make the captures explicit

trait_selection_prlf_defined_with_sub = the lifetime `{$sub_symbol}` defined here...
trait_selection_prlf_defined_without_sub = the lifetime defined here...
trait_selection_prlf_known_limitation = this is a known limitation that will be removed in the future (see issue #100013 <https://github.com/rust-lang/rust/issues/100013> for more information)

trait_selection_prlf_must_outlive_with_sup = ...must outlive the lifetime `{$sup_symbol}` defined here
trait_selection_prlf_must_outlive_without_sup = ...must outlive the lifetime defined here
trait_selection_reborrow = ...so that reference does not outlive borrowed content
trait_selection_ref_longer_than_data = in type `{$ty}`, reference has a longer lifetime than the data it references

trait_selection_reference_outlives_referent = ...so that the reference type `{$name}` does not outlive the data it points at
trait_selection_region_explanation = {$pref_kind ->
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
    [defined_here_reg] the lifetime `{$desc_arg}` as defined here
}{$suff_kind ->
    *[should_not_happen] [{$suff_kind}]
    [empty]{""}
    [continues] ...
    [req_by_binding] {" "}as required by this binding
}

trait_selection_relate_object_bound = ...so that it can be closed over into an object
trait_selection_relate_param_bound = ...so that the type `{$name}` will meet its required lifetime bounds{$continues ->
    [true] ...
    *[false] {""}
}
trait_selection_relate_param_bound_2 = ...that is required by this bound
trait_selection_relate_region_param_bound = ...so that the declared lifetime parameter bounds are satisfied
trait_selection_ril_because_of = because of this returned expression
trait_selection_ril_introduced_by = requirement introduced by this return type
trait_selection_ril_introduced_here = `'static` requirement introduced here
trait_selection_ril_static_introduced_by = "`'static` lifetime requirement introduced by the return type

trait_selection_rustc_on_unimplemented_empty_on_clause = empty `on`-clause in `#[rustc_on_unimplemented]`
    .label = empty `on`-clause here
trait_selection_rustc_on_unimplemented_expected_identifier = expected an identifier inside this `on`-clause
    .label = expected an identifier here, not `{$path}`
trait_selection_rustc_on_unimplemented_expected_one_predicate_in_not = expected a single predicate in `not(..)`
    .label = unexpected quantity of predicates here
trait_selection_rustc_on_unimplemented_invalid_flag = invalid flag in `on`-clause
    .label = expected one of the `crate_local`, `direct` or `from_desugaring` flags, not `{$invalid_flag}`
trait_selection_rustc_on_unimplemented_invalid_name = invalid name in `on`-clause
    .label = expected one of `cause`, `from_desugaring`, `Self` or any generic parameter of the trait, not `{$invalid_name}`
trait_selection_rustc_on_unimplemented_invalid_predicate = this predicate is invalid
    .label = expected one of `any`, `all` or `not` here, not `{$invalid_pred}`
trait_selection_rustc_on_unimplemented_missing_value = this attribute must have a value
    .label = expected value here
    .note = e.g. `#[rustc_on_unimplemented(message="foo")]`
trait_selection_rustc_on_unimplemented_unsupported_literal_in_on = literals inside `on`-clauses are not supported
    .label = unexpected literal here

trait_selection_source_kind_closure_return =
    try giving this closure an explicit return type

# coroutine_kind  may need to be translated
trait_selection_source_kind_fully_qualified =
    try using a fully qualified path to specify the expected types

trait_selection_source_kind_subdiag_generic_label =
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

trait_selection_source_kind_subdiag_generic_suggestion =
    consider specifying the generic {$arg_count ->
    [one] argument
    *[other] arguments
    }

trait_selection_source_kind_subdiag_let = {$kind ->
    [with_pattern] consider giving `{$name}` an explicit type
    [closure] consider giving this closure parameter an explicit type
    *[other] consider giving this pattern a type
}{$x_kind ->
    [has_name] , where the {$prefix_kind ->
        *[type] type for {$prefix}
        [const_with_param] value of const parameter
        [const] value of the constant
    } `{$arg_name}` is specified
    [underscore] , where the placeholders `_` are specified
    *[empty] {""}
}

trait_selection_srs_add = consider returning the local binding `{$ident}`
trait_selection_srs_add_one = consider returning one of these bindings

trait_selection_srs_remove = consider removing this semicolon
trait_selection_srs_remove_and_box = consider removing this semicolon and boxing the expressions
trait_selection_stp_wrap_many = try wrapping the pattern in a variant of `{$path}`

trait_selection_stp_wrap_one = try wrapping the pattern in `{$variant}`
trait_selection_subtype = ...so that the {$requirement ->
    [method_compat] method type is compatible with trait
    [type_compat] associated type is compatible with trait
    [const_compat] const is compatible with trait
    [expr_assignable] expression is assignable
    [if_else_different] `if` and `else` have incompatible types
    [no_else] `if` missing an `else` returns `()`
    [fn_main_correct_type] `main` function has the correct type
    [fn_lang_correct_type] lang item function has the correct type
    [intrinsic_correct_type] intrinsic has the correct type
    [method_correct_type] method receiver has the correct type
    *[other] types are compatible
}
trait_selection_subtype_2 = ...so that {$requirement ->
    [method_compat] method type is compatible with trait
    [type_compat] associated type is compatible with trait
    [const_compat] const is compatible with trait
    [expr_assignable] expression is assignable
    [if_else_different] `if` and `else` have incompatible types
    [no_else] `if` missing an `else` returns `()`
    [fn_main_correct_type] `main` function has the correct type
    [fn_lang_correct_type] lang item function has the correct type
    [intrinsic_correct_type] intrinsic has the correct type
    [method_correct_type] method receiver has the correct type
    *[other] types are compatible
}

trait_selection_suggest_accessing_field = you might have meant to use field `{$name}` whose type is `{$ty}`

trait_selection_suggest_add_let_for_letchains = consider adding `let`

trait_selection_tid_consider_borrowing = consider borrowing this type parameter in the trait
trait_selection_tid_param_help = the lifetime requirements from the `impl` do not correspond to the requirements in the `trait`

trait_selection_tid_rel_help = verify the lifetime relationships in the `trait` and `impl` between the `self` argument, the other inputs and its output
trait_selection_trait_has_no_impls = this trait has no implementations, consider adding one

trait_selection_trait_impl_diff = `impl` item signature doesn't match `trait` item signature
    .found = found `{$found}`
    .expected = expected `{$expected}`
    .expected_found = expected signature `{$expected}`
               {"   "}found signature `{$found}`

trait_selection_trait_placeholder_mismatch = implementation of `{$trait_def_id}` is not general enough
    .label_satisfy = doesn't satisfy where-clause
    .label_where = due to a where-clause on `{$def_id}`...
    .label_dup = implementation of `{$trait_def_id}` is not general enough

trait_selection_try_cannot_convert = `?` operator cannot convert from `{$found}` to `{$expected}`

trait_selection_tuple_trailing_comma = use a trailing comma to create a tuple with one element

trait_selection_ty_alias_overflow = in case this is a recursive type alias, consider using a struct, enum, or union instead
trait_selection_type_annotations_needed = {$source_kind ->
    [closure] type annotations needed for the closure `{$source_name}`
    [normal] type annotations needed for `{$source_name}`
    *[other] type annotations needed
}
    .label = type must be known at this point

trait_selection_types_declared_different = these two types are declared with different lifetimes...

trait_selection_unable_to_construct_constant_value = unable to construct a constant value for the unevaluated constant {$unevaluated}

trait_selection_unknown_format_parameter_for_on_unimplemented_attr = there is no parameter `{$argument_name}` on trait `{$trait_name}`
    .help = expect either a generic argument name or {"`{Self}`"} as format argument

trait_selection_warn_removing_apit_params_for_overcapture = you could use a `use<...>` bound to explicitly specify captures, but argument-position `impl Trait`s are not nameable

trait_selection_warn_removing_apit_params_for_undercapture = you could use a `use<...>` bound to explicitly capture `{$new_lifetime}`, but argument-position `impl Trait`s are not nameable

trait_selection_where_copy_predicates = copy the `where` clause predicates from the trait

trait_selection_where_remove = remove the `where` clause
trait_selection_wrapped_parser_error = {$description}
    .label = {$label}
