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
    [fn_start_correct_type] #[start]` function has the correct type
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
    [fn_start_correct_type] #[start]` function has the correct type
    [intristic_correct_type] intrinsic has the correct type
    [method_correct_type] method receiver has the correct type
    *[other] types are compatible
}

infer_reborrow = ...so that reference does not outlive borrowed content
infer_reborrow_upvar = ...so that closure can access `{$name}`
infer_relate_object_bound = ...so that it can be closed over into an object
infer_data_borrowed = ...so that the type `{$name}` is not borrowed for too long
infer_reference_outlives_referent = ...so that the reference type `{$name}` does not outlive the data it points at
infer_relate_param_bound = ...so that the type `{$name}` will meet its required lifetime bounds{$continues ->
    [true] ...
    *[false] {""}
}
infer_relate_param_bound_2 = ...that is required by this bound
infer_relate_region_param_bound = ...so that the declared lifetime parameter bounds are satisfied
infer_compare_impl_item_obligation = ...so that the definition in impl matches the definition from the trait

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
    [true] -> {" "}from `{$label_var1}`
    *[false] -> {""}
} flows{$label_var2_exists ->
    [true] -> {" "}into `{$label_var2}`
    *[false] -> {""}
} here

infer_lifetime_param_suggestion = consider introducing a named lifetime parameter{$is_impl ->
    [true] {" "}and update trait if needed
    *[false] {""}
}
infer_lifetime_param_suggestion_elided = each elided lifetime in input position becomes a distinct lifetime

infer_region_explanation = {$pref_kind ->
    *[should_not_happen] [{$pref_kind}]
    [empty] {""}
}{$pref_kind ->
    [empty] {""}
    *[other] {" "}
}{$desc_kind ->
    *[should_not_happen] [{$desc_kind}]
    [restatic] the static lifetime
    [reempty] the empty lifetime
    [reemptyuni] the empty lifetime in universe {$desc_arg}
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
}

infer_mismatched_static_lifetime = incompatible lifetime on type
infer_msl_impl_note = ...does not necessarily outlive the static lifetime introduced by the compatible `impl`
infer_msl_introduces_static = introduces a `'static` lifetime requirement
infer_msl_unmet_req = because this has an unmet lifetime requirement
infer_msl_trait_note = this has an implicit `'static` lifetime requirement
infer_msl_trait_sugg = consider relaxing the implicit `'static` requirement
