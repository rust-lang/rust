mod ok {
    use crate::tests::*;
    #[test]
    fn anonymous_const() {
        run_and_expect_no_errors("test_data/parser/inline/ok/anonymous_const.rs");
    }
    #[test]
    fn arb_self_types() {
        run_and_expect_no_errors("test_data/parser/inline/ok/arb_self_types.rs");
    }
    #[test]
    fn arg_with_attr() { run_and_expect_no_errors("test_data/parser/inline/ok/arg_with_attr.rs"); }
    #[test]
    fn array_attrs() { run_and_expect_no_errors("test_data/parser/inline/ok/array_attrs.rs"); }
    #[test]
    fn array_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/array_expr.rs"); }
    #[test]
    fn array_type() { run_and_expect_no_errors("test_data/parser/inline/ok/array_type.rs"); }
    #[test]
    fn as_precedence() { run_and_expect_no_errors("test_data/parser/inline/ok/as_precedence.rs"); }
    #[test]
    fn asm_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/asm_expr.rs"); }
    #[test]
    fn asm_label() { run_and_expect_no_errors("test_data/parser/inline/ok/asm_label.rs"); }
    #[test]
    fn assoc_const_eq() {
        run_and_expect_no_errors("test_data/parser/inline/ok/assoc_const_eq.rs");
    }
    #[test]
    fn assoc_item_list() {
        run_and_expect_no_errors("test_data/parser/inline/ok/assoc_item_list.rs");
    }
    #[test]
    fn assoc_item_list_inner_attrs() {
        run_and_expect_no_errors("test_data/parser/inline/ok/assoc_item_list_inner_attrs.rs");
    }
    #[test]
    fn assoc_type_bound() {
        run_and_expect_no_errors("test_data/parser/inline/ok/assoc_type_bound.rs");
    }
    #[test]
    fn assoc_type_eq() { run_and_expect_no_errors("test_data/parser/inline/ok/assoc_type_eq.rs"); }
    #[test]
    fn async_trait_bound() {
        run_and_expect_no_errors("test_data/parser/inline/ok/async_trait_bound.rs");
    }
    #[test]
    fn attr_on_expr_stmt() {
        run_and_expect_no_errors("test_data/parser/inline/ok/attr_on_expr_stmt.rs");
    }
    #[test]
    fn await_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/await_expr.rs"); }
    #[test]
    fn bare_dyn_types_with_leading_lifetime() {
        run_and_expect_no_errors(
            "test_data/parser/inline/ok/bare_dyn_types_with_leading_lifetime.rs",
        );
    }
    #[test]
    fn become_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/become_expr.rs"); }
    #[test]
    fn bind_pat() { run_and_expect_no_errors("test_data/parser/inline/ok/bind_pat.rs"); }
    #[test]
    fn binop_resets_statementness() {
        run_and_expect_no_errors("test_data/parser/inline/ok/binop_resets_statementness.rs");
    }
    #[test]
    fn block() { run_and_expect_no_errors("test_data/parser/inline/ok/block.rs"); }
    #[test]
    fn block_items() { run_and_expect_no_errors("test_data/parser/inline/ok/block_items.rs"); }
    #[test]
    fn box_pat() { run_and_expect_no_errors("test_data/parser/inline/ok/box_pat.rs"); }
    #[test]
    fn break_ambiguity() {
        run_and_expect_no_errors("test_data/parser/inline/ok/break_ambiguity.rs");
    }
    #[test]
    fn break_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/break_expr.rs"); }
    #[test]
    fn builtin_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/builtin_expr.rs"); }
    #[test]
    fn call_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/call_expr.rs"); }
    #[test]
    fn cast_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/cast_expr.rs"); }
    #[test]
    fn closure_binder() {
        run_and_expect_no_errors("test_data/parser/inline/ok/closure_binder.rs");
    }
    #[test]
    fn closure_body_underscore_assignment() {
        run_and_expect_no_errors(
            "test_data/parser/inline/ok/closure_body_underscore_assignment.rs",
        );
    }
    #[test]
    fn closure_params() {
        run_and_expect_no_errors("test_data/parser/inline/ok/closure_params.rs");
    }
    #[test]
    fn closure_range_method_call() {
        run_and_expect_no_errors("test_data/parser/inline/ok/closure_range_method_call.rs");
    }
    #[test]
    fn const_arg() { run_and_expect_no_errors("test_data/parser/inline/ok/const_arg.rs"); }
    #[test]
    fn const_arg_block() {
        run_and_expect_no_errors("test_data/parser/inline/ok/const_arg_block.rs");
    }
    #[test]
    fn const_arg_bool_literal() {
        run_and_expect_no_errors("test_data/parser/inline/ok/const_arg_bool_literal.rs");
    }
    #[test]
    fn const_arg_literal() {
        run_and_expect_no_errors("test_data/parser/inline/ok/const_arg_literal.rs");
    }
    #[test]
    fn const_arg_negative_number() {
        run_and_expect_no_errors("test_data/parser/inline/ok/const_arg_negative_number.rs");
    }
    #[test]
    fn const_block_pat() {
        run_and_expect_no_errors("test_data/parser/inline/ok/const_block_pat.rs");
    }
    #[test]
    fn const_closure() { run_and_expect_no_errors("test_data/parser/inline/ok/const_closure.rs"); }
    #[test]
    fn const_item() { run_and_expect_no_errors("test_data/parser/inline/ok/const_item.rs"); }
    #[test]
    fn const_param() { run_and_expect_no_errors("test_data/parser/inline/ok/const_param.rs"); }
    #[test]
    fn const_param_default_expression() {
        run_and_expect_no_errors("test_data/parser/inline/ok/const_param_default_expression.rs");
    }
    #[test]
    fn const_param_default_literal() {
        run_and_expect_no_errors("test_data/parser/inline/ok/const_param_default_literal.rs");
    }
    #[test]
    fn const_param_default_path() {
        run_and_expect_no_errors("test_data/parser/inline/ok/const_param_default_path.rs");
    }
    #[test]
    fn const_trait_bound() {
        run_and_expect_no_errors("test_data/parser/inline/ok/const_trait_bound.rs");
    }
    #[test]
    fn const_where_clause() {
        run_and_expect_no_errors("test_data/parser/inline/ok/const_where_clause.rs");
    }
    #[test]
    fn continue_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/continue_expr.rs"); }
    #[test]
    fn crate_path() { run_and_expect_no_errors("test_data/parser/inline/ok/crate_path.rs"); }
    #[test]
    fn crate_visibility() {
        run_and_expect_no_errors("test_data/parser/inline/ok/crate_visibility.rs");
    }
    #[test]
    fn crate_visibility_in() {
        run_and_expect_no_errors("test_data/parser/inline/ok/crate_visibility_in.rs");
    }
    #[test]
    fn default_async_fn() {
        run_and_expect_no_errors("test_data/parser/inline/ok/default_async_fn.rs");
    }
    #[test]
    fn default_async_unsafe_fn() {
        run_and_expect_no_errors("test_data/parser/inline/ok/default_async_unsafe_fn.rs");
    }
    #[test]
    fn default_item() { run_and_expect_no_errors("test_data/parser/inline/ok/default_item.rs"); }
    #[test]
    fn default_unsafe_item() {
        run_and_expect_no_errors("test_data/parser/inline/ok/default_unsafe_item.rs");
    }
    #[test]
    fn destructuring_assignment_struct_rest_pattern() {
        run_and_expect_no_errors(
            "test_data/parser/inline/ok/destructuring_assignment_struct_rest_pattern.rs",
        );
    }
    #[test]
    fn destructuring_assignment_wildcard_pat() {
        run_and_expect_no_errors(
            "test_data/parser/inline/ok/destructuring_assignment_wildcard_pat.rs",
        );
    }
    #[test]
    fn dot_dot_pat() { run_and_expect_no_errors("test_data/parser/inline/ok/dot_dot_pat.rs"); }
    #[test]
    fn dyn_trait_type() {
        run_and_expect_no_errors("test_data/parser/inline/ok/dyn_trait_type.rs");
    }
    #[test]
    fn dyn_trait_type_weak() {
        run_and_expect_no_errors_with_edition(
            "test_data/parser/inline/ok/dyn_trait_type_weak.rs",
            crate::Edition::Edition2015,
        );
    }
    #[test]
    fn edition_2015_dyn_prefix_inside_generic_arg() {
        run_and_expect_no_errors_with_edition(
            "test_data/parser/inline/ok/edition_2015_dyn_prefix_inside_generic_arg.rs",
            crate::Edition::Edition2015,
        );
    }
    #[test]
    fn effect_blocks() { run_and_expect_no_errors("test_data/parser/inline/ok/effect_blocks.rs"); }
    #[test]
    fn exclusive_range_pat() {
        run_and_expect_no_errors("test_data/parser/inline/ok/exclusive_range_pat.rs");
    }
    #[test]
    fn expr_literals() { run_and_expect_no_errors("test_data/parser/inline/ok/expr_literals.rs"); }
    #[test]
    fn expression_after_block() {
        run_and_expect_no_errors("test_data/parser/inline/ok/expression_after_block.rs");
    }
    #[test]
    fn extern_block() { run_and_expect_no_errors("test_data/parser/inline/ok/extern_block.rs"); }
    #[test]
    fn extern_crate() { run_and_expect_no_errors("test_data/parser/inline/ok/extern_crate.rs"); }
    #[test]
    fn extern_crate_rename() {
        run_and_expect_no_errors("test_data/parser/inline/ok/extern_crate_rename.rs");
    }
    #[test]
    fn field_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/field_expr.rs"); }
    #[test]
    fn fn_() { run_and_expect_no_errors("test_data/parser/inline/ok/fn_.rs"); }
    #[test]
    fn fn_decl() { run_and_expect_no_errors("test_data/parser/inline/ok/fn_decl.rs"); }
    #[test]
    fn fn_def_param() { run_and_expect_no_errors("test_data/parser/inline/ok/fn_def_param.rs"); }
    #[test]
    fn fn_pointer_param_ident_path() {
        run_and_expect_no_errors("test_data/parser/inline/ok/fn_pointer_param_ident_path.rs");
    }
    #[test]
    fn fn_pointer_type() {
        run_and_expect_no_errors("test_data/parser/inline/ok/fn_pointer_type.rs");
    }
    #[test]
    fn fn_pointer_type_with_ret() {
        run_and_expect_no_errors("test_data/parser/inline/ok/fn_pointer_type_with_ret.rs");
    }
    #[test]
    fn fn_pointer_unnamed_arg() {
        run_and_expect_no_errors("test_data/parser/inline/ok/fn_pointer_unnamed_arg.rs");
    }
    #[test]
    fn for_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/for_expr.rs"); }
    #[test]
    fn for_range_from() {
        run_and_expect_no_errors("test_data/parser/inline/ok/for_range_from.rs");
    }
    #[test]
    fn for_type() { run_and_expect_no_errors("test_data/parser/inline/ok/for_type.rs"); }
    #[test]
    fn full_range_expr() {
        run_and_expect_no_errors("test_data/parser/inline/ok/full_range_expr.rs");
    }
    #[test]
    fn function_ret_type() {
        run_and_expect_no_errors("test_data/parser/inline/ok/function_ret_type.rs");
    }
    #[test]
    fn function_type_params() {
        run_and_expect_no_errors("test_data/parser/inline/ok/function_type_params.rs");
    }
    #[test]
    fn function_where_clause() {
        run_and_expect_no_errors("test_data/parser/inline/ok/function_where_clause.rs");
    }
    #[test]
    fn gen_blocks() {
        run_and_expect_no_errors_with_edition(
            "test_data/parser/inline/ok/gen_blocks.rs",
            crate::Edition::Edition2024,
        );
    }
    #[test]
    fn generic_arg() { run_and_expect_no_errors("test_data/parser/inline/ok/generic_arg.rs"); }
    #[test]
    fn generic_arg_bounds() {
        run_and_expect_no_errors("test_data/parser/inline/ok/generic_arg_bounds.rs");
    }
    #[test]
    fn generic_const() { run_and_expect_no_errors("test_data/parser/inline/ok/generic_const.rs"); }
    #[test]
    fn generic_param_attribute() {
        run_and_expect_no_errors("test_data/parser/inline/ok/generic_param_attribute.rs");
    }
    #[test]
    fn generic_param_list() {
        run_and_expect_no_errors("test_data/parser/inline/ok/generic_param_list.rs");
    }
    #[test]
    fn half_open_range_pat() {
        run_and_expect_no_errors("test_data/parser/inline/ok/half_open_range_pat.rs");
    }
    #[test]
    fn if_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/if_expr.rs"); }
    #[test]
    fn impl_item() { run_and_expect_no_errors("test_data/parser/inline/ok/impl_item.rs"); }
    #[test]
    fn impl_item_const() {
        run_and_expect_no_errors("test_data/parser/inline/ok/impl_item_const.rs");
    }
    #[test]
    fn impl_item_neg() { run_and_expect_no_errors("test_data/parser/inline/ok/impl_item_neg.rs"); }
    #[test]
    fn impl_trait_type() {
        run_and_expect_no_errors("test_data/parser/inline/ok/impl_trait_type.rs");
    }
    #[test]
    fn impl_type_params() {
        run_and_expect_no_errors("test_data/parser/inline/ok/impl_type_params.rs");
    }
    #[test]
    fn index_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/index_expr.rs"); }
    #[test]
    fn label() { run_and_expect_no_errors("test_data/parser/inline/ok/label.rs"); }
    #[test]
    fn labeled_block() { run_and_expect_no_errors("test_data/parser/inline/ok/labeled_block.rs"); }
    #[test]
    fn lambda_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/lambda_expr.rs"); }
    #[test]
    fn lambda_ret_block() {
        run_and_expect_no_errors("test_data/parser/inline/ok/lambda_ret_block.rs");
    }
    #[test]
    fn let_else() { run_and_expect_no_errors("test_data/parser/inline/ok/let_else.rs"); }
    #[test]
    fn let_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/let_expr.rs"); }
    #[test]
    fn let_stmt() { run_and_expect_no_errors("test_data/parser/inline/ok/let_stmt.rs"); }
    #[test]
    fn let_stmt_ascription() {
        run_and_expect_no_errors("test_data/parser/inline/ok/let_stmt_ascription.rs");
    }
    #[test]
    fn let_stmt_init() { run_and_expect_no_errors("test_data/parser/inline/ok/let_stmt_init.rs"); }
    #[test]
    fn lifetime_arg() { run_and_expect_no_errors("test_data/parser/inline/ok/lifetime_arg.rs"); }
    #[test]
    fn lifetime_param() {
        run_and_expect_no_errors("test_data/parser/inline/ok/lifetime_param.rs");
    }
    #[test]
    fn literal_pattern() {
        run_and_expect_no_errors("test_data/parser/inline/ok/literal_pattern.rs");
    }
    #[test]
    fn loop_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/loop_expr.rs"); }
    #[test]
    fn macro_call_type() {
        run_and_expect_no_errors("test_data/parser/inline/ok/macro_call_type.rs");
    }
    #[test]
    fn macro_def() { run_and_expect_no_errors("test_data/parser/inline/ok/macro_def.rs"); }
    #[test]
    fn macro_def_curly() {
        run_and_expect_no_errors("test_data/parser/inline/ok/macro_def_curly.rs");
    }
    #[test]
    fn macro_inside_generic_arg() {
        run_and_expect_no_errors("test_data/parser/inline/ok/macro_inside_generic_arg.rs");
    }
    #[test]
    fn macro_rules_as_macro_name() {
        run_and_expect_no_errors("test_data/parser/inline/ok/macro_rules_as_macro_name.rs");
    }
    #[test]
    fn macro_rules_non_brace() {
        run_and_expect_no_errors("test_data/parser/inline/ok/macro_rules_non_brace.rs");
    }
    #[test]
    fn marco_pat() { run_and_expect_no_errors("test_data/parser/inline/ok/marco_pat.rs"); }
    #[test]
    fn match_arm() { run_and_expect_no_errors("test_data/parser/inline/ok/match_arm.rs"); }
    #[test]
    fn match_arms_commas() {
        run_and_expect_no_errors("test_data/parser/inline/ok/match_arms_commas.rs");
    }
    #[test]
    fn match_arms_inner_attribute() {
        run_and_expect_no_errors("test_data/parser/inline/ok/match_arms_inner_attribute.rs");
    }
    #[test]
    fn match_arms_outer_attributes() {
        run_and_expect_no_errors("test_data/parser/inline/ok/match_arms_outer_attributes.rs");
    }
    #[test]
    fn match_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/match_expr.rs"); }
    #[test]
    fn match_guard() { run_and_expect_no_errors("test_data/parser/inline/ok/match_guard.rs"); }
    #[test]
    fn metas() { run_and_expect_no_errors("test_data/parser/inline/ok/metas.rs"); }
    #[test]
    fn method_call_expr() {
        run_and_expect_no_errors("test_data/parser/inline/ok/method_call_expr.rs");
    }
    #[test]
    fn mod_contents() { run_and_expect_no_errors("test_data/parser/inline/ok/mod_contents.rs"); }
    #[test]
    fn mod_item() { run_and_expect_no_errors("test_data/parser/inline/ok/mod_item.rs"); }
    #[test]
    fn mod_item_curly() {
        run_and_expect_no_errors("test_data/parser/inline/ok/mod_item_curly.rs");
    }
    #[test]
    fn never_type() { run_and_expect_no_errors("test_data/parser/inline/ok/never_type.rs"); }
    #[test]
    fn no_dyn_trait_leading_for() {
        run_and_expect_no_errors("test_data/parser/inline/ok/no_dyn_trait_leading_for.rs");
    }
    #[test]
    fn no_semi_after_block() {
        run_and_expect_no_errors("test_data/parser/inline/ok/no_semi_after_block.rs");
    }
    #[test]
    fn nocontentexpr() { run_and_expect_no_errors("test_data/parser/inline/ok/nocontentexpr.rs"); }
    #[test]
    fn nocontentexpr_after_item() {
        run_and_expect_no_errors("test_data/parser/inline/ok/nocontentexpr_after_item.rs");
    }
    #[test]
    fn offset_of_parens() {
        run_and_expect_no_errors("test_data/parser/inline/ok/offset_of_parens.rs");
    }
    #[test]
    fn or_pattern() { run_and_expect_no_errors("test_data/parser/inline/ok/or_pattern.rs"); }
    #[test]
    fn param_list() { run_and_expect_no_errors("test_data/parser/inline/ok/param_list.rs"); }
    #[test]
    fn param_list_vararg() {
        run_and_expect_no_errors("test_data/parser/inline/ok/param_list_vararg.rs");
    }
    #[test]
    fn param_outer_arg() {
        run_and_expect_no_errors("test_data/parser/inline/ok/param_outer_arg.rs");
    }
    #[test]
    fn paren_type() { run_and_expect_no_errors("test_data/parser/inline/ok/paren_type.rs"); }
    #[test]
    fn path_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/path_expr.rs"); }
    #[test]
    fn path_fn_trait_args() {
        run_and_expect_no_errors("test_data/parser/inline/ok/path_fn_trait_args.rs");
    }
    #[test]
    fn path_part() { run_and_expect_no_errors("test_data/parser/inline/ok/path_part.rs"); }
    #[test]
    fn path_type() { run_and_expect_no_errors("test_data/parser/inline/ok/path_type.rs"); }
    #[test]
    fn path_type_with_bounds() {
        run_and_expect_no_errors("test_data/parser/inline/ok/path_type_with_bounds.rs");
    }
    #[test]
    fn placeholder_pat() {
        run_and_expect_no_errors("test_data/parser/inline/ok/placeholder_pat.rs");
    }
    #[test]
    fn placeholder_type() {
        run_and_expect_no_errors("test_data/parser/inline/ok/placeholder_type.rs");
    }
    #[test]
    fn pointer_type_mut() {
        run_and_expect_no_errors("test_data/parser/inline/ok/pointer_type_mut.rs");
    }
    #[test]
    fn postfix_range() { run_and_expect_no_errors("test_data/parser/inline/ok/postfix_range.rs"); }
    #[test]
    fn precise_capturing() {
        run_and_expect_no_errors("test_data/parser/inline/ok/precise_capturing.rs");
    }
    #[test]
    fn pub_parens_typepath() {
        run_and_expect_no_errors("test_data/parser/inline/ok/pub_parens_typepath.rs");
    }
    #[test]
    fn pub_tuple_field() {
        run_and_expect_no_errors("test_data/parser/inline/ok/pub_tuple_field.rs");
    }
    #[test]
    fn qual_paths() { run_and_expect_no_errors("test_data/parser/inline/ok/qual_paths.rs"); }
    #[test]
    fn question_for_type_trait_bound() {
        run_and_expect_no_errors("test_data/parser/inline/ok/question_for_type_trait_bound.rs");
    }
    #[test]
    fn range_pat() { run_and_expect_no_errors("test_data/parser/inline/ok/range_pat.rs"); }
    #[test]
    fn record_field_attrs() {
        run_and_expect_no_errors("test_data/parser/inline/ok/record_field_attrs.rs");
    }
    #[test]
    fn record_field_default_values() {
        run_and_expect_no_errors("test_data/parser/inline/ok/record_field_default_values.rs");
    }
    #[test]
    fn record_field_list() {
        run_and_expect_no_errors("test_data/parser/inline/ok/record_field_list.rs");
    }
    #[test]
    fn record_field_pat_leading_or() {
        run_and_expect_no_errors("test_data/parser/inline/ok/record_field_pat_leading_or.rs");
    }
    #[test]
    fn record_lit() { run_and_expect_no_errors("test_data/parser/inline/ok/record_lit.rs"); }
    #[test]
    fn record_literal_field_with_attr() {
        run_and_expect_no_errors("test_data/parser/inline/ok/record_literal_field_with_attr.rs");
    }
    #[test]
    fn record_pat_field() {
        run_and_expect_no_errors("test_data/parser/inline/ok/record_pat_field.rs");
    }
    #[test]
    fn record_pat_field_list() {
        run_and_expect_no_errors("test_data/parser/inline/ok/record_pat_field_list.rs");
    }
    #[test]
    fn ref_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/ref_expr.rs"); }
    #[test]
    fn ref_pat() { run_and_expect_no_errors("test_data/parser/inline/ok/ref_pat.rs"); }
    #[test]
    fn reference_type() {
        run_and_expect_no_errors("test_data/parser/inline/ok/reference_type.rs");
    }
    #[test]
    fn return_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/return_expr.rs"); }
    #[test]
    fn return_type_syntax_in_path() {
        run_and_expect_no_errors("test_data/parser/inline/ok/return_type_syntax_in_path.rs");
    }
    #[test]
    fn safe_outside_of_extern() {
        run_and_expect_no_errors("test_data/parser/inline/ok/safe_outside_of_extern.rs");
    }
    #[test]
    fn self_param() { run_and_expect_no_errors("test_data/parser/inline/ok/self_param.rs"); }
    #[test]
    fn self_param_outer_attr() {
        run_and_expect_no_errors("test_data/parser/inline/ok/self_param_outer_attr.rs");
    }
    #[test]
    fn singleton_tuple_type() {
        run_and_expect_no_errors("test_data/parser/inline/ok/singleton_tuple_type.rs");
    }
    #[test]
    fn slice_pat() { run_and_expect_no_errors("test_data/parser/inline/ok/slice_pat.rs"); }
    #[test]
    fn slice_type() { run_and_expect_no_errors("test_data/parser/inline/ok/slice_type.rs"); }
    #[test]
    fn stmt_bin_expr_ambiguity() {
        run_and_expect_no_errors("test_data/parser/inline/ok/stmt_bin_expr_ambiguity.rs");
    }
    #[test]
    fn stmt_postfix_expr_ambiguity() {
        run_and_expect_no_errors("test_data/parser/inline/ok/stmt_postfix_expr_ambiguity.rs");
    }
    #[test]
    fn struct_initializer_with_defaults() {
        run_and_expect_no_errors("test_data/parser/inline/ok/struct_initializer_with_defaults.rs");
    }
    #[test]
    fn struct_item() { run_and_expect_no_errors("test_data/parser/inline/ok/struct_item.rs"); }
    #[test]
    fn trait_alias() { run_and_expect_no_errors("test_data/parser/inline/ok/trait_alias.rs"); }
    #[test]
    fn trait_alias_where_clause() {
        run_and_expect_no_errors("test_data/parser/inline/ok/trait_alias_where_clause.rs");
    }
    #[test]
    fn trait_item() { run_and_expect_no_errors("test_data/parser/inline/ok/trait_item.rs"); }
    #[test]
    fn trait_item_bounds() {
        run_and_expect_no_errors("test_data/parser/inline/ok/trait_item_bounds.rs");
    }
    #[test]
    fn trait_item_generic_params() {
        run_and_expect_no_errors("test_data/parser/inline/ok/trait_item_generic_params.rs");
    }
    #[test]
    fn trait_item_where_clause() {
        run_and_expect_no_errors("test_data/parser/inline/ok/trait_item_where_clause.rs");
    }
    #[test]
    fn try_block_expr() {
        run_and_expect_no_errors("test_data/parser/inline/ok/try_block_expr.rs");
    }
    #[test]
    fn try_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/try_expr.rs"); }
    #[test]
    fn try_macro_fallback() {
        run_and_expect_no_errors_with_edition(
            "test_data/parser/inline/ok/try_macro_fallback.rs",
            crate::Edition::Edition2015,
        );
    }
    #[test]
    fn try_macro_rules() {
        run_and_expect_no_errors_with_edition(
            "test_data/parser/inline/ok/try_macro_rules.rs",
            crate::Edition::Edition2015,
        );
    }
    #[test]
    fn tuple_attrs() { run_and_expect_no_errors("test_data/parser/inline/ok/tuple_attrs.rs"); }
    #[test]
    fn tuple_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/tuple_expr.rs"); }
    #[test]
    fn tuple_field_attrs() {
        run_and_expect_no_errors("test_data/parser/inline/ok/tuple_field_attrs.rs");
    }
    #[test]
    fn tuple_pat() { run_and_expect_no_errors("test_data/parser/inline/ok/tuple_pat.rs"); }
    #[test]
    fn tuple_pat_fields() {
        run_and_expect_no_errors("test_data/parser/inline/ok/tuple_pat_fields.rs");
    }
    #[test]
    fn tuple_struct() { run_and_expect_no_errors("test_data/parser/inline/ok/tuple_struct.rs"); }
    #[test]
    fn tuple_struct_where() {
        run_and_expect_no_errors("test_data/parser/inline/ok/tuple_struct_where.rs");
    }
    #[test]
    fn type_alias() { run_and_expect_no_errors("test_data/parser/inline/ok/type_alias.rs"); }
    #[test]
    fn type_item_type_params() {
        run_and_expect_no_errors("test_data/parser/inline/ok/type_item_type_params.rs");
    }
    #[test]
    fn type_item_where_clause() {
        run_and_expect_no_errors("test_data/parser/inline/ok/type_item_where_clause.rs");
    }
    #[test]
    fn type_item_where_clause_deprecated() {
        run_and_expect_no_errors("test_data/parser/inline/ok/type_item_where_clause_deprecated.rs");
    }
    #[test]
    fn type_param() { run_and_expect_no_errors("test_data/parser/inline/ok/type_param.rs"); }
    #[test]
    fn type_param_bounds() {
        run_and_expect_no_errors("test_data/parser/inline/ok/type_param_bounds.rs");
    }
    #[test]
    fn type_param_default() {
        run_and_expect_no_errors("test_data/parser/inline/ok/type_param_default.rs");
    }
    #[test]
    fn type_path_in_pattern() {
        run_and_expect_no_errors("test_data/parser/inline/ok/type_path_in_pattern.rs");
    }
    #[test]
    fn typepathfn_with_coloncolon() {
        run_and_expect_no_errors("test_data/parser/inline/ok/typepathfn_with_coloncolon.rs");
    }
    #[test]
    fn unary_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/unary_expr.rs"); }
    #[test]
    fn union_item() { run_and_expect_no_errors("test_data/parser/inline/ok/union_item.rs"); }
    #[test]
    fn unit_struct() { run_and_expect_no_errors("test_data/parser/inline/ok/unit_struct.rs"); }
    #[test]
    fn unit_type() { run_and_expect_no_errors("test_data/parser/inline/ok/unit_type.rs"); }
    #[test]
    fn use_item() { run_and_expect_no_errors("test_data/parser/inline/ok/use_item.rs"); }
    #[test]
    fn use_tree() { run_and_expect_no_errors("test_data/parser/inline/ok/use_tree.rs"); }
    #[test]
    fn use_tree_abs_star() {
        run_and_expect_no_errors("test_data/parser/inline/ok/use_tree_abs_star.rs");
    }
    #[test]
    fn use_tree_alias() {
        run_and_expect_no_errors("test_data/parser/inline/ok/use_tree_alias.rs");
    }
    #[test]
    fn use_tree_list() { run_and_expect_no_errors("test_data/parser/inline/ok/use_tree_list.rs"); }
    #[test]
    fn use_tree_path() { run_and_expect_no_errors("test_data/parser/inline/ok/use_tree_path.rs"); }
    #[test]
    fn use_tree_path_star() {
        run_and_expect_no_errors("test_data/parser/inline/ok/use_tree_path_star.rs");
    }
    #[test]
    fn use_tree_path_use_tree() {
        run_and_expect_no_errors("test_data/parser/inline/ok/use_tree_path_use_tree.rs");
    }
    #[test]
    fn use_tree_star() { run_and_expect_no_errors("test_data/parser/inline/ok/use_tree_star.rs"); }
    #[test]
    fn variant_discriminant() {
        run_and_expect_no_errors("test_data/parser/inline/ok/variant_discriminant.rs");
    }
    #[test]
    fn where_clause() { run_and_expect_no_errors("test_data/parser/inline/ok/where_clause.rs"); }
    #[test]
    fn where_pred_for() {
        run_and_expect_no_errors("test_data/parser/inline/ok/where_pred_for.rs");
    }
    #[test]
    fn while_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/while_expr.rs"); }
    #[test]
    fn yeet_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/yeet_expr.rs"); }
    #[test]
    fn yield_expr() { run_and_expect_no_errors("test_data/parser/inline/ok/yield_expr.rs"); }
}
mod err {
    use crate::tests::*;
    #[test]
    fn angled_path_without_qual() {
        run_and_expect_errors("test_data/parser/inline/err/angled_path_without_qual.rs");
    }
    #[test]
    fn anonymous_static() {
        run_and_expect_errors("test_data/parser/inline/err/anonymous_static.rs");
    }
    #[test]
    fn arg_list_recovery() {
        run_and_expect_errors("test_data/parser/inline/err/arg_list_recovery.rs");
    }
    #[test]
    fn array_type_missing_semi() {
        run_and_expect_errors("test_data/parser/inline/err/array_type_missing_semi.rs");
    }
    #[test]
    fn async_without_semicolon() {
        run_and_expect_errors("test_data/parser/inline/err/async_without_semicolon.rs");
    }
    #[test]
    fn bad_asm_expr() { run_and_expect_errors("test_data/parser/inline/err/bad_asm_expr.rs"); }
    #[test]
    fn comma_after_default_values_syntax() {
        run_and_expect_errors("test_data/parser/inline/err/comma_after_default_values_syntax.rs");
    }
    #[test]
    fn comma_after_functional_update_syntax() {
        run_and_expect_errors(
            "test_data/parser/inline/err/comma_after_functional_update_syntax.rs",
        );
    }
    #[test]
    fn crate_visibility_empty_recover() {
        run_and_expect_errors("test_data/parser/inline/err/crate_visibility_empty_recover.rs");
    }
    #[test]
    fn empty_param_slot() {
        run_and_expect_errors("test_data/parser/inline/err/empty_param_slot.rs");
    }
    #[test]
    fn empty_segment() { run_and_expect_errors("test_data/parser/inline/err/empty_segment.rs"); }
    #[test]
    fn fn_pointer_type_missing_fn() {
        run_and_expect_errors("test_data/parser/inline/err/fn_pointer_type_missing_fn.rs");
    }
    #[test]
    fn gen_fn() {
        run_and_expect_errors_with_edition(
            "test_data/parser/inline/err/gen_fn.rs",
            crate::Edition::Edition2021,
        );
    }
    #[test]
    fn generic_arg_list_recover() {
        run_and_expect_errors("test_data/parser/inline/err/generic_arg_list_recover.rs");
    }
    #[test]
    fn generic_arg_list_recover_expr() {
        run_and_expect_errors("test_data/parser/inline/err/generic_arg_list_recover_expr.rs");
    }
    #[test]
    fn generic_param_list_recover() {
        run_and_expect_errors("test_data/parser/inline/err/generic_param_list_recover.rs");
    }
    #[test]
    fn generic_static() { run_and_expect_errors("test_data/parser/inline/err/generic_static.rs"); }
    #[test]
    fn impl_type() { run_and_expect_errors("test_data/parser/inline/err/impl_type.rs"); }
    #[test]
    fn let_else_right_curly_brace() {
        run_and_expect_errors("test_data/parser/inline/err/let_else_right_curly_brace.rs");
    }
    #[test]
    fn macro_rules_as_macro_name() {
        run_and_expect_errors("test_data/parser/inline/err/macro_rules_as_macro_name.rs");
    }
    #[test]
    fn match_arms_recovery() {
        run_and_expect_errors("test_data/parser/inline/err/match_arms_recovery.rs");
    }
    #[test]
    fn meta_recovery() { run_and_expect_errors("test_data/parser/inline/err/meta_recovery.rs"); }
    #[test]
    fn method_call_missing_argument_list() {
        run_and_expect_errors("test_data/parser/inline/err/method_call_missing_argument_list.rs");
    }
    #[test]
    fn misplaced_label_err() {
        run_and_expect_errors("test_data/parser/inline/err/misplaced_label_err.rs");
    }
    #[test]
    fn missing_fn_param_type() {
        run_and_expect_errors("test_data/parser/inline/err/missing_fn_param_type.rs");
    }
    #[test]
    fn path_item_without_excl() {
        run_and_expect_errors("test_data/parser/inline/err/path_item_without_excl.rs");
    }
    #[test]
    fn pointer_type_no_mutability() {
        run_and_expect_errors("test_data/parser/inline/err/pointer_type_no_mutability.rs");
    }
    #[test]
    fn precise_capturing_invalid() {
        run_and_expect_errors("test_data/parser/inline/err/precise_capturing_invalid.rs");
    }
    #[test]
    fn pub_expr() { run_and_expect_errors("test_data/parser/inline/err/pub_expr.rs"); }
    #[test]
    fn record_literal_before_ellipsis_recovery() {
        run_and_expect_errors(
            "test_data/parser/inline/err/record_literal_before_ellipsis_recovery.rs",
        );
    }
    #[test]
    fn record_literal_field_eq_recovery() {
        run_and_expect_errors("test_data/parser/inline/err/record_literal_field_eq_recovery.rs");
    }
    #[test]
    fn record_literal_missing_ellipsis_recovery() {
        run_and_expect_errors(
            "test_data/parser/inline/err/record_literal_missing_ellipsis_recovery.rs",
        );
    }
    #[test]
    fn record_pat_field_eq_recovery() {
        run_and_expect_errors("test_data/parser/inline/err/record_pat_field_eq_recovery.rs");
    }
    #[test]
    fn recover_from_missing_assoc_item_binding() {
        run_and_expect_errors(
            "test_data/parser/inline/err/recover_from_missing_assoc_item_binding.rs",
        );
    }
    #[test]
    fn recover_from_missing_const_default() {
        run_and_expect_errors("test_data/parser/inline/err/recover_from_missing_const_default.rs");
    }
    #[test]
    fn static_where_clause() {
        run_and_expect_errors("test_data/parser/inline/err/static_where_clause.rs");
    }
    #[test]
    fn struct_field_recover() {
        run_and_expect_errors("test_data/parser/inline/err/struct_field_recover.rs");
    }
    #[test]
    fn top_level_let() { run_and_expect_errors("test_data/parser/inline/err/top_level_let.rs"); }
    #[test]
    fn tuple_expr_leading_comma() {
        run_and_expect_errors("test_data/parser/inline/err/tuple_expr_leading_comma.rs");
    }
    #[test]
    fn tuple_field_list_recovery() {
        run_and_expect_errors("test_data/parser/inline/err/tuple_field_list_recovery.rs");
    }
    #[test]
    fn tuple_pat_leading_comma() {
        run_and_expect_errors("test_data/parser/inline/err/tuple_pat_leading_comma.rs");
    }
    #[test]
    fn type_bounds_macro_call_recovery() {
        run_and_expect_errors("test_data/parser/inline/err/type_bounds_macro_call_recovery.rs");
    }
    #[test]
    fn type_in_array_recover() {
        run_and_expect_errors("test_data/parser/inline/err/type_in_array_recover.rs");
    }
    #[test]
    fn unsafe_block_in_mod() {
        run_and_expect_errors("test_data/parser/inline/err/unsafe_block_in_mod.rs");
    }
    #[test]
    fn use_tree_list_err_recovery() {
        run_and_expect_errors("test_data/parser/inline/err/use_tree_list_err_recovery.rs");
    }
}
