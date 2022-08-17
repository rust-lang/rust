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
