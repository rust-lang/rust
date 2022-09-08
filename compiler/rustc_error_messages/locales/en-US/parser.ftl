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
    .note = unlike in e.g., python and PHP, `&&` and `||` are used for logical operators
    .use_amp_amp_for_conjunction = use `&&` to perform logical conjunction
    .use_pipe_pipe_for_disjunction = use `||` to perform logical disjunction

parser_tilde_is_not_unary_operator = `~` cannot be used as a unary operator
    .suggestion = use `!` to perform bitwise not

parser_unexpected_token_after_not = unexpected {$negated_desc} after identifier
    .suggestion = use `!` to perform logical negation

parser_malformed_loop_label = malformed loop label
    .suggestion = use the correct loop label format

parser_lifetime_in_borrow_expression = borrow expressions cannot be annotated with lifetimes
    .suggestion = remove the lifetime annotation
    .label = annotated with lifetime here

parser_field_expression_with_generic = field expressions cannot have generic arguments

parser_macro_invocation_with_qualified_path = macros cannot use qualified paths

parser_unexpected_token_after_label = expected `while`, `for`, `loop` or `{"{"}` after a label

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
