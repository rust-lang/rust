//! See test_utils/src/marks.rs

test_utils::marks![
    option_order
    introduce_var_in_comment_is_not_applicable
    test_introduce_var_expr_stmt
    test_introduce_var_last_expr
    not_applicable_outside_of_bind_pat
    test_not_inline_mut_variable
    test_not_applicable_if_variable_unused
    change_visibility_field_false_positive
    test_add_from_impl_already_exists
];
