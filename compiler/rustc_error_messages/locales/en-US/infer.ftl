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

infer_need_type_info_in_generator =
    type inside {$generator_kind} must be known in this context
