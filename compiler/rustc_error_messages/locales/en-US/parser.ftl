parser_struct_literal_body_without_path =
    struct literal body without path
    .suggestion = you might have forgotten to add the struct literal inside the block

parser_maybe_report_ambiguous_plus =
    ambiguous `+` in a type
    .suggestion = use parentheses to disambiguate

parser_maybe_recover_from_bad_type_plus =
    expected a path on the left-hand side of `+`, not `{$ty}`

parser_add_paren = try adding parentheses

parser_forgot_paren = perhaps you forgot parentheses?

parser_expect_path = expected a path

parser_maybe_recover_from_bad_qpath_stage_2 =
    missing angle brackets in associated item path
    .suggestion = try: `{$ty}`

parser_incorrect_semicolon =
    expected item, found `;`
    .suggestion = remove this semicolon
    .help = {$name} declarations are not followed by a semicolon

parser_incorrect_use_of_await =
    incorrect use of `await`
    .parentheses_suggestion = `await` is not a method call, remove the parentheses
    .postfix_suggestion = `await` is a postfix operation

parser_in_in_typo =
    expected iterable, found keyword `in`
    .suggestion = remove the duplicated `in`

parser_invalid_variable_declaration =
    invalid variable declaration

parser_switch_mut_let_order =
    switch the order of `mut` and `let`
parser_missing_let_before_mut = missing keyword
parser_use_let_not_auto = write `let` instead of `auto` to introduce a new variable
parser_use_let_not_var = write `let` instead of `var` to introduce a new variable

parser_invalid_comparison_operator = invalid comparison operator `{$invalid}`
    .use_instead = `{$invalid}` is not a valid comparison operator, use `{$correct}`
    .spaceship_operator_invalid = `<=>` is not a valid comparison operator, use `std::cmp::Ordering`

parser_invalid_logical_operator = `{$incorrect}` is not a logical operator
    .note = unlike in e.g., Python and PHP, `&&` and `||` are used for logical operators
    .use_amp_amp_for_conjunction = use `&&` to perform logical conjunction
    .use_pipe_pipe_for_disjunction = use `||` to perform logical disjunction

parser_tilde_is_not_unary_operator = `~` cannot be used as a unary operator
    .suggestion = use `!` to perform bitwise not

parser_unexpected_token_after_not = unexpected {$negated_desc} after identifier
parser_unexpected_token_after_not_bitwise = use `!` to perform bitwise not
parser_unexpected_token_after_not_logical = use `!` to perform logical negation
parser_unexpected_token_after_not_default = use `!` to perform logical negation or bitwise not

parser_malformed_loop_label = malformed loop label
    .suggestion = use the correct loop label format

parser_lifetime_in_borrow_expression = borrow expressions cannot be annotated with lifetimes
    .suggestion = remove the lifetime annotation
    .label = annotated with lifetime here

parser_field_expression_with_generic = field expressions cannot have generic arguments

parser_macro_invocation_with_qualified_path = macros cannot use qualified paths

parser_unexpected_token_after_label = expected `while`, `for`, `loop` or `{"{"}` after a label
    .suggestion_remove_label = consider removing the label
    .suggestion_enclose_in_block = consider enclosing expression in a block

parser_require_colon_after_labeled_expression = labeled expression must be followed by `:`
    .note = labels are used before loops and blocks, allowing e.g., `break 'label` to them
    .label = the label
    .suggestion = add `:` after the label

parser_do_catch_syntax_removed = found removed `do catch` syntax
    .note = following RFC #2388, the new non-placeholder syntax is `try`
    .suggestion = replace with the new syntax

parser_float_literal_requires_integer_part = float literals must have an integer part
    .suggestion = must have an integer part

parser_invalid_int_literal_width = invalid width `{$width}` for integer literal
    .help = valid widths are 8, 16, 32, 64 and 128

parser_invalid_num_literal_base_prefix = invalid base prefix for number literal
    .note = base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase
    .suggestion = try making the prefix lowercase

parser_invalid_num_literal_suffix = invalid suffix `{$suffix}` for number literal
    .label = invalid suffix `{$suffix}`
    .help = the suffix must be one of the numeric types (`u32`, `isize`, `f32`, etc.)

parser_invalid_float_literal_width = invalid width `{$width}` for float literal
    .help = valid widths are 32 and 64

parser_invalid_float_literal_suffix = invalid suffix `{$suffix}` for float literal
    .label = invalid suffix `{$suffix}`
    .help = valid suffixes are `f32` and `f64`

parser_int_literal_too_large = integer literal is too large

parser_missing_semicolon_before_array = expected `;`, found `[`
    .suggestion = consider adding `;` here

parser_invalid_block_macro_segment = cannot use a `block` macro fragment here
    .label = the `block` fragment is within this context

parser_if_expression_missing_then_block = this `if` expression is missing a block after the condition
    .add_then_block = add a block here
    .condition_possibly_unfinished = this binary operation is possibly unfinished

parser_if_expression_missing_condition = missing condition for `if` expression
    .condition_label = expected condition here
    .block_label = if this block is the condition of the `if` expression, then it must be followed by another block

parser_expected_expression_found_let = expected expression, found `let` statement

parser_expected_else_block = expected `{"{"}`, found {$first_tok}
    .label = expected an `if` or a block after this `else`
    .suggestion = add an `if` if this is the condition of a chained `else if` statement

parser_outer_attribute_not_allowed_on_if_else = outer attributes are not allowed on `if` and `else` branches
    .branch_label = the attributes are attached to this branch
    .ctx_label = the branch belongs to this `{$ctx}`
    .suggestion = remove the attributes

parser_missing_in_in_for_loop = missing `in` in `for` loop
    .use_in_not_of = try using `in` here instead
    .add_in = try adding `in` here

parser_missing_comma_after_match_arm = expected `,` following `match` arm
    .suggestion = missing a comma here to end this `match` arm

parser_catch_after_try = keyword `catch` cannot follow a `try` block
    .help = try using `match` on the result of the `try` block instead

parser_comma_after_base_struct = cannot use a comma after the base struct
    .note = the base struct must always be the last field
    .suggestion = remove this comma

parser_eq_field_init = expected `:`, found `=`
    .suggestion = replace equals symbol with a colon

parser_dotdotdot = unexpected token: `...`
    .suggest_exclusive_range = use `..` for an exclusive range
    .suggest_inclusive_range = or `..=` for an inclusive range

parser_left_arrow_operator = unexpected token: `<-`
    .suggestion = if you meant to write a comparison against a negative value, add a space in between `<` and `-`

parser_remove_let = expected pattern, found `let`
    .suggestion = remove the unnecessary `let` keyword

parser_use_eq_instead = unexpected `==`
    .suggestion = try using `=` instead

parser_use_empty_block_not_semi = expected { "`{}`" }, found `;`
    .suggestion = try using { "`{}`" } instead

parser_comparison_interpreted_as_generic =
    `<` is interpreted as a start of generic arguments for `{$type}`, not a comparison
    .label_args = interpreted as generic arguments
    .label_comparison = not interpreted as comparison
    .suggestion = try comparing the cast value

parser_shift_interpreted_as_generic =
    `<<` is interpreted as a start of generic arguments for `{$type}`, not a shift
    .label_args = interpreted as generic arguments
    .label_comparison = not interpreted as shift
    .suggestion = try shifting the cast value

parser_found_expr_would_be_stmt = expected expression, found `{$token}`
    .label = expected expression

parser_leading_plus_not_supported = leading `+` is not supported
    .label = unexpected `+`
    .suggestion_remove_plus = try removing the `+`

parser_parentheses_with_struct_fields = invalid `struct` delimiters or `fn` call arguments
    .suggestion_braces_for_struct = if `{$type}` is a struct, use braces as delimiters
    .suggestion_no_fields_for_fn = if `{$type}` is a function, use the arguments directly

parser_labeled_loop_in_break = parentheses are required around this expression to avoid confusion with a labeled break expression

parser_sugg_wrap_expression_in_parentheses = wrap the expression in parentheses

parser_array_brackets_instead_of_braces = this is a block expression, not an array
    .suggestion = to make an array, use square brackets instead of curly braces

parser_match_arm_body_without_braces = `match` arm body without braces
    .label_statements = {$num_statements ->
            [one] this statement is not surrounded by a body
           *[other] these statements are not surrounded by a body
        }
    .label_arrow = while parsing the `match` arm starting here
    .suggestion_add_braces = surround the {$num_statements ->
            [one] statement
           *[other] statements
        } with a body
    .suggestion_use_comma_not_semicolon = use a comma to end a `match` arm expression

parser_struct_literal_not_allowed_here = struct literals are not allowed here
    .suggestion = surround the struct literal with parentheses

parser_invalid_interpolated_expression = invalid interpolated expression

parser_hexadecimal_float_literal_not_supported = hexadecimal float literal is not supported
parser_octal_float_literal_not_supported = octal float literal is not supported
parser_binary_float_literal_not_supported = binary float literal is not supported
parser_not_supported = not supported

parser_invalid_literal_suffix = suffixes on {$kind} literals are invalid
    .label = invalid suffix `{$suffix}`

parser_invalid_literal_suffix_on_tuple_index = suffixes on a tuple index are invalid
    .label = invalid suffix `{$suffix}`
    .tuple_exception_line_1 = `{$suffix}` is *temporarily* accepted on tuple index fields as it was incorrectly accepted on stable for a few releases
    .tuple_exception_line_2 = on proc macros, you'll want to use `syn::Index::from` or `proc_macro::Literal::*_unsuffixed` for code that will desugar to tuple field access
    .tuple_exception_line_3 = see issue #60210 <https://github.com/rust-lang/rust/issues/60210> for more information

parser_non_string_abi_literal = non-string ABI literal
    .suggestion = specify the ABI with a string literal

parser_mismatched_closing_delimiter = mismatched closing delimiter: `{$delimiter}`
    .label_unmatched = mismatched closing delimiter
    .label_opening_candidate = closing delimiter possibly meant for this
    .label_unclosed = unclosed delimiter

parser_incorrect_visibility_restriction = incorrect visibility restriction
    .help = some possible visibility restrictions are:
            `pub(crate)`: visible only on the current crate
            `pub(super)`: visible only in the current module's parent
            `pub(in path::to::module)`: visible only on the specified path
    .suggestion = make this visible only to module `{$inner_str}` with `in`

parser_assignment_else_not_allowed = <assignment> ... else {"{"} ... {"}"} is not allowed

parser_expected_statement_after_outer_attr = expected statement after outer attribute

parser_doc_comment_does_not_document_anything = found a documentation comment that doesn't document anything
    .help = doc comments must come before what they document, if a comment was intended use `//`
    .suggestion = missing comma here

parser_const_let_mutually_exclusive = `const` and `let` are mutually exclusive
    .suggestion = remove `let`

parser_invalid_expression_in_let_else = a `{$operator}` expression cannot be directly assigned in `let...else`
parser_invalid_curly_in_let_else = right curly brace `{"}"}` before `else` in a `let...else` statement not allowed

parser_compound_assignment_expression_in_let = can't reassign to an uninitialized variable
    .suggestion = initialize the variable
    .help = if you meant to overwrite, remove the `let` binding

parser_suffixed_literal_in_attribute = suffixed literals are not allowed in attributes
    .help = instead of using a suffixed literal (`1u8`, `1.0f32`, etc.), use an unsuffixed version (`1`, `1.0`, etc.)

parser_invalid_meta_item = expected unsuffixed literal or identifier, found `{$token}`

parser_label_inner_attr_does_not_annotate_this = the inner attribute doesn't annotate this {$item}
parser_sugg_change_inner_attr_to_outer = to annotate the {$item}, change the attribute from inner to outer style

parser_inner_attr_not_permitted_after_outer_doc_comment = an inner attribute is not permitted following an outer doc comment
    .label_attr = not permitted following an outer doc comment
    .label_prev_doc_comment = previous doc comment
    .label_does_not_annotate_this = {parser_label_inner_attr_does_not_annotate_this}
    .sugg_change_inner_to_outer = {parser_sugg_change_inner_attr_to_outer}

parser_inner_attr_not_permitted_after_outer_attr = an inner attribute is not permitted following an outer attribute
    .label_attr = not permitted following an outer attribute
    .label_prev_attr = previous outer attribute
    .label_does_not_annotate_this = {parser_label_inner_attr_does_not_annotate_this}
    .sugg_change_inner_to_outer = {parser_sugg_change_inner_attr_to_outer}

parser_inner_attr_not_permitted = an inner attribute is not permitted in this context
    .label_does_not_annotate_this = {parser_label_inner_attr_does_not_annotate_this}
    .sugg_change_inner_to_outer = {parser_sugg_change_inner_attr_to_outer}

parser_inner_attr_explanation = inner attributes, like `#![no_std]`, annotate the item enclosing them, and are usually found at the beginning of source files
parser_outer_attr_explanation = outer attributes, like `#[test]`, annotate the item following them

parser_inner_doc_comment_not_permitted = expected outer doc comment
    .note = inner doc comments like this (starting with `//!` or `/*!`) can only appear before items
    .suggestion = you might have meant to write a regular comment
    .label_does_not_annotate_this = the inner doc comment doesn't annotate this {$item}
    .sugg_change_inner_to_outer = to annotate the {$item}, change the doc comment from inner to outer style

parser_expected_identifier_found_reserved_identifier_str = expected identifier, found reserved identifier `{$token}`
parser_expected_identifier_found_keyword_str = expected identifier, found keyword `{$token}`
parser_expected_identifier_found_reserved_keyword_str = expected identifier, found reserved keyword `{$token}`
parser_expected_identifier_found_doc_comment_str = expected identifier, found doc comment `{$token}`
parser_expected_identifier_found_str = expected identifier, found `{$token}`

parser_expected_identifier_found_reserved_identifier = expected identifier, found reserved identifier
parser_expected_identifier_found_keyword = expected identifier, found keyword
parser_expected_identifier_found_reserved_keyword = expected identifier, found reserved keyword
parser_expected_identifier_found_doc_comment = expected identifier, found doc comment
parser_expected_identifier = expected identifier

parser_sugg_escape_to_use_as_identifier = escape `{$ident_name}` to use it as an identifier

parser_sugg_remove_comma = remove this comma

parser_expected_semi_found_reserved_identifier_str = expected `;`, found reserved identifier `{$token}`
parser_expected_semi_found_keyword_str = expected `;`, found keyword `{$token}`
parser_expected_semi_found_reserved_keyword_str = expected `;`, found reserved keyword `{$token}`
parser_expected_semi_found_doc_comment_str = expected `;`, found doc comment `{$token}`
parser_expected_semi_found_str = expected `;`, found `{$token}`

parser_sugg_change_this_to_semi = change this to `;`
parser_sugg_add_semi = add `;` here
parser_label_unexpected_token = unexpected token

parser_unmatched_angle_brackets = {$num_extra_brackets ->
        [one] unmatched angle bracket
       *[other] unmatched angle brackets
    }
    .suggestion = {$num_extra_brackets ->
            [one] remove extra angle bracket
           *[other] remove extra angle brackets
        }

parser_generic_parameters_without_angle_brackets = generic parameters without surrounding angle brackets
    .suggestion = surround the type parameters with angle brackets

parser_comparison_operators_cannot_be_chained = comparison operators cannot be chained
    .sugg_parentheses_for_function_args = or use `(...)` if you meant to specify fn arguments
    .sugg_split_comparison = split the comparison into two
    .sugg_parenthesize = parenthesize the comparison
parser_sugg_turbofish_syntax = use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments

parser_question_mark_in_type = invalid `?` in type
    .label = `?` is only allowed on expressions, not types
    .suggestion = if you meant to express that the type might not contain a value, use the `Option` wrapper type

parser_unexpected_parentheses_in_for_head = unexpected parentheses surrounding `for` loop head
    .suggestion = remove parentheses in `for` loop

parser_doc_comment_on_param_type = documentation comments cannot be applied to a function parameter's type
    .label = doc comments are not allowed here

parser_attribute_on_param_type = attributes cannot be applied to a function parameter's type
    .label = attributes are not allowed here

parser_pattern_method_param_without_body = patterns aren't allowed in methods without bodies
    .suggestion = give this argument a name or use an underscore to ignore it

parser_self_param_not_first = unexpected `self` parameter in function
    .label = must be the first parameter of an associated function

parser_const_generic_without_braces = expressions must be enclosed in braces to be used as const generic arguments
    .suggestion = enclose the `const` expression in braces

parser_unexpected_const_param_declaration = unexpected `const` parameter declaration
    .label = expected a `const` expression, not a parameter declaration
    .suggestion = `const` parameters must be declared for the `impl`

parser_unexpected_const_in_generic_param = expected lifetime, type, or constant, found keyword `const`
    .suggestion = the `const` keyword is only needed in the definition of the type

parser_async_move_order_incorrect = the order of `move` and `async` is incorrect
    .suggestion = try switching the order

parser_double_colon_in_bound = expected `:` followed by trait or lifetime
    .suggestion = use single colon
