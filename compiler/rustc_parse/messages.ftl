parse_add_paren = try adding parentheses

parse_ambiguous_range_pattern = the range pattern here has ambiguous interpretation
parse_ambiguous_range_pattern_suggestion = add parentheses to clarify the precedence

parse_array_brackets_instead_of_braces = this is a block expression, not an array
    .suggestion = to make an array, use square brackets instead of curly braces

parse_array_index_offset_of = array indexing not supported in offset_of

parse_asm_expected_comma = expected token: `,`
    .label = expected `,`

parse_asm_expected_other = expected operand, {$is_inline_asm ->
    [false] options
    *[true] clobber_abi, options
    }, or additional template string

parse_asm_expected_register_class_or_explicit_register = expected register class or explicit register

parse_asm_expected_string_literal = expected string literal
    .label = not a string literal

parse_asm_non_abi = at least one abi must be provided as an argument to `clobber_abi`

parse_asm_requires_template = requires at least a template string argument

parse_asm_sym_no_path = expected a path for argument to `sym`

parse_asm_underscore_input = _ cannot be used for input operands

parse_asm_unsupported_operand = the `{$symbol}` operand cannot be used with `{$macro_name}!`
    .label = the `{$symbol}` operand is not meaningful for global-scoped inline assembly, remove it

parse_assignment_else_not_allowed = <assignment> ... else {"{"} ... {"}"} is not allowed

parse_associated_static_item_not_allowed = associated `static` items are not allowed

parse_async_block_in_2015 = `async` blocks are only allowed in Rust 2018 or later

parse_async_bound_modifier_in_2015 = `async` trait bounds are only allowed in Rust 2018 or later

parse_async_fn_in_2015 = `async fn` is not permitted in Rust 2015
    .label = to use `async fn`, switch to Rust 2018 or later

parse_async_impl = `async` trait implementations are unsupported

parse_async_move_block_in_2015 = `async move` blocks are only allowed in Rust 2018 or later

parse_async_move_order_incorrect = the order of `move` and `async` is incorrect
    .suggestion = try switching the order

parse_async_use_block_in_2015 = `async use` blocks are only allowed in Rust 2018 or later

parse_async_use_order_incorrect = the order of `use` and `async` is incorrect
    .suggestion = try switching the order

parse_at_dot_dot_in_struct_pattern = `@ ..` is not supported in struct patterns
    .suggestion = bind to each field separately or, if you don't need them, just remove `{$ident} @`

parse_at_in_struct_pattern = Unexpected `@` in struct pattern
    .note = struct patterns use `field: pattern` syntax to bind to fields
    .help = consider replacing `new_name @ field_name` with `field_name: new_name` if that is what you intended

parse_attr_after_generic = trailing attribute after generic parameter
    .label = attributes must go before parameters

parse_attr_without_generics = attribute without generic parameters
    .label = attributes are only permitted when preceding parameters

parse_attribute_on_param_type = attributes cannot be applied to a function parameter's type
    .label = attributes are not allowed here

parse_bad_assoc_type_bounds = bounds on associated types do not belong here
    .label = belongs in `where` clause

parse_bad_item_kind = {$descr} is not supported in {$ctx}
    .help = consider moving the {$descr} out to a nearby module scope

parse_bad_return_type_notation_output =
    return type not allowed with return type notation
    .suggestion = remove the return type

parse_bare_cr = {$double_quotes ->
    [true] bare CR not allowed in string, use `\r` instead
    *[false] character constant must be escaped: `\r`
    }
    .escape = escape the character

parse_bare_cr_in_raw_string = bare CR not allowed in raw string

parse_binder_and_polarity = `for<...>` binder not allowed with `{$polarity}` trait polarity modifier
    .label = there is not a well-defined meaning for a higher-ranked `{$polarity}` trait

parse_binder_before_modifiers = `for<...>` binder should be placed before trait bound modifiers
    .label = place the `for<...>` binder before any modifiers

parse_bounds_not_allowed_on_trait_aliases = bounds are not allowed on trait aliases

parse_box_not_pat = expected pattern, found {$descr}
    .note = `box` is a reserved keyword
    .suggestion = escape `box` to use it as an identifier

parse_box_syntax_removed = `box_syntax` has been removed
parse_box_syntax_removed_suggestion = use `Box::new()` instead

parse_cannot_be_raw_ident = `{$ident}` cannot be a raw identifier

parse_cannot_be_raw_lifetime = `{$ident}` cannot be a raw lifetime

parse_catch_after_try = keyword `catch` cannot follow a `try` block
    .help = try using `match` on the result of the `try` block instead

parse_cfg_attr_bad_delim = wrong `cfg_attr` delimiters
parse_colon_as_semi = statements are terminated with a semicolon
    .suggestion = use a semicolon instead

parse_comma_after_base_struct = cannot use a comma after the base struct
    .note = the base struct must always be the last field
    .suggestion = remove this comma

parse_comparison_interpreted_as_generic =
    `<` is interpreted as a start of generic arguments for `{$type}`, not a comparison
    .label_args = interpreted as generic arguments
    .label_comparison = not interpreted as comparison
    .suggestion = try comparing the cast value

parse_comparison_operators_cannot_be_chained = comparison operators cannot be chained
    .sugg_parentheses_for_function_args = or use `(...)` if you meant to specify fn arguments
    .sugg_split_comparison = split the comparison into two
    .sugg_parenthesize = parenthesize the comparison
parse_compound_assignment_expression_in_let = can't reassign to an uninitialized variable
    .suggestion = initialize the variable
    .help = if you meant to overwrite, remove the `let` binding

parse_const_generic_without_braces = expressions must be enclosed in braces to be used as const generic arguments
    .suggestion = enclose the `const` expression in braces

parse_const_global_cannot_be_mutable = const globals cannot be mutable
    .label = cannot be mutable
    .suggestion = you might want to declare a static instead

parse_const_let_mutually_exclusive = `const` and `let` are mutually exclusive
    .suggestion = remove `let`

parse_cr_doc_comment = bare CR not allowed in {$block ->
    [true] block doc-comment
    *[false] doc-comment
}

parse_default_not_followed_by_item = `default` is not followed by an item
    .label = the `default` qualifier
    .note = only `fn`, `const`, `type`, or `impl` items may be prefixed by `default`

parse_do_catch_syntax_removed = found removed `do catch` syntax
    .note = following RFC #2388, the new non-placeholder syntax is `try`
    .suggestion = replace with the new syntax

parse_doc_comment_does_not_document_anything = found a documentation comment that doesn't document anything
    .help = doc comments must come before what they document, if a comment was intended use `//`
    .suggestion = missing comma here

parse_doc_comment_on_param_type = documentation comments cannot be applied to a function parameter's type
    .label = doc comments are not allowed here

parse_dot_dot_dot_for_remaining_fields = expected field pattern, found `{$token_str}`
    .suggestion = to omit remaining fields, use `..`

parse_dot_dot_dot_range_to_pattern_not_allowed = range-to patterns with `...` are not allowed
    .suggestion = use `..=` instead

parse_dot_dot_range_attribute = attributes are not allowed on range expressions starting with `..`

parse_dotdotdot = unexpected token: `...`
    .suggest_exclusive_range = use `..` for an exclusive range
    .suggest_inclusive_range = or `..=` for an inclusive range

parse_dotdotdot_rest_pattern = unexpected `...`
    .label = not a valid pattern
    .suggestion = for a rest pattern, use `..` instead of `...`

parse_double_colon_in_bound = expected `:` followed by trait or lifetime
    .suggestion = use single colon

parse_dyn_after_mut = `mut` must precede `dyn`
    .suggestion = place `mut` before `dyn`

parse_empty_exponent_float = expected at least one digit in exponent

parse_empty_unicode_escape = empty unicode escape
    .label = this escape must have at least 1 hex digit

parse_enum_pattern_instead_of_identifier = expected identifier, found enum pattern

parse_enum_struct_mutually_exclusive = `enum` and `struct` are mutually exclusive
    .suggestion = replace `enum struct` with

parse_eq_field_init = expected `:`, found `=`
    .suggestion = replace equals symbol with a colon

parse_escape_only_char = {$byte ->
    [true] byte
    *[false] character
    } constant must be escaped: `{$escaped_msg}`
    .escape = escape the character

parse_expect_dotdot_not_dotdotdot = expected `..`, found `...`
    .suggestion = use `..` to fill in the rest of the fields

parse_expect_eq_instead_of_eqeq = expected `=`, found `==`
    .suggestion = consider using `=` here

parse_expect_label_found_ident = expected a label, found an identifier
    .suggestion = labels start with a tick

parse_expect_path = expected a path

parse_expected_binding_left_of_at = left-hand side of `@` must be a binding
    .label_lhs = interpreted as a pattern, not a binding
    .label_rhs = also a pattern
    .note = bindings are `x`, `mut x`, `ref x`, and `ref mut x`

parse_expected_builtin_ident = expected identifier after `builtin #`

parse_expected_comma_after_pattern_field = expected `,`

parse_expected_else_block = expected `{"{"}`, found {$first_tok}
    .label = expected an `if` or a block after this `else`
    .suggestion = add an `if` if this is the condition of a chained `else if` statement

parse_expected_expression_found_let = expected expression, found `let` statement
    .note = only supported directly in conditions of `if` and `while` expressions
    .not_supported_or = `||` operators are not supported in let chain expressions
    .not_supported_parentheses = `let`s wrapped in parentheses are not supported in a context with let chains

parse_expected_fn_path_found_fn_keyword = expected identifier, found keyword `fn`
    .suggestion = use `Fn` to refer to the trait

parse_expected_identifier = expected identifier

parse_expected_identifier_found_doc_comment = expected identifier, found doc comment
parse_expected_identifier_found_doc_comment_str = expected identifier, found doc comment `{$token}`
parse_expected_identifier_found_keyword = expected identifier, found keyword
parse_expected_identifier_found_keyword_str = expected identifier, found keyword `{$token}`
parse_expected_identifier_found_metavar = expected identifier, found metavariable
# This one deliberately doesn't print a token.
parse_expected_identifier_found_metavar_str = expected identifier, found metavariable
parse_expected_identifier_found_reserved_identifier = expected identifier, found reserved identifier
parse_expected_identifier_found_reserved_identifier_str = expected identifier, found reserved identifier `{$token}`
parse_expected_identifier_found_reserved_keyword = expected identifier, found reserved keyword
parse_expected_identifier_found_reserved_keyword_str = expected identifier, found reserved keyword `{$token}`
parse_expected_identifier_found_str = expected identifier, found `{$token}`

parse_expected_mut_or_const_in_raw_pointer_type = expected `mut` or `const` keyword in raw pointer type
    .suggestion = add `mut` or `const` here

parse_expected_semi_found_doc_comment_str = expected `;`, found doc comment `{$token}`
parse_expected_semi_found_keyword_str = expected `;`, found keyword `{$token}`
# This one deliberately doesn't print a token.
parse_expected_semi_found_metavar_str = expected `;`, found metavariable
parse_expected_semi_found_reserved_identifier_str = expected `;`, found reserved identifier `{$token}`
parse_expected_semi_found_reserved_keyword_str = expected `;`, found reserved keyword `{$token}`
parse_expected_semi_found_str = expected `;`, found `{$token}`

parse_expected_statement_after_outer_attr = expected statement after outer attribute

parse_expected_struct_field = expected one of `,`, `:`, or `{"}"}`, found `{$token}`
    .label = expected one of `,`, `:`, or `{"}"}`
    .ident_label = while parsing this struct field

parse_expected_trait_in_trait_impl_found_type = expected a trait, found type

parse_expr_rarrow_call = `->` is not valid syntax for field accesses and method calls
    .suggestion = try using `.` instead
    .help = the `.` operator will automatically dereference the value, except if the value is a raw pointer

parse_extern_crate_name_with_dashes = crate name using dashes are not valid in `extern crate` statements
    .label = dash-separated idents are not valid
    .suggestion = if the original crate name uses dashes you need to use underscores in the code

parse_extern_item_cannot_be_const = extern items cannot be `const`
    .suggestion = try using a static value
    .note = for more information, visit https://doc.rust-lang.org/std/keyword.extern.html

parse_extra_if_in_let_else = remove the `if` if you meant to write a `let...else` statement

parse_extra_impl_keyword_in_trait_impl = unexpected `impl` keyword
    .suggestion = remove the extra `impl`
    .note = this is parsed as an `impl Trait` type, but a trait is expected at this position


parse_field_expression_with_generic = field expressions cannot have generic arguments

parse_float_literal_requires_integer_part = float literals must have an integer part
    .suggestion = must have an integer part

parse_float_literal_unsupported_base = {$base} float literal is not supported

parse_fn_pointer_cannot_be_async = an `fn` pointer type cannot be `async`
    .label = `async` because of this
    .suggestion = remove the `async` qualifier
    .note = allowed qualifiers are: `unsafe` and `extern`

parse_fn_pointer_cannot_be_const = an `fn` pointer type cannot be `const`
    .label = `const` because of this
    .suggestion = remove the `const` qualifier
    .note = allowed qualifiers are: `unsafe` and `extern`

parse_fn_ptr_with_generics = function pointer types may not have generic parameters
    .suggestion = consider moving the lifetime {$arity ->
        [one] parameter
        *[other] parameters
    } to {$for_param_list_exists ->
        [true] the
        *[false] a
    } `for` parameter list

parse_fn_trait_missing_paren = `Fn` bounds require arguments in parentheses
    .add_paren = add the missing parentheses

parse_forgot_paren = perhaps you forgot parentheses?

parse_found_expr_would_be_stmt = expected expression, found `{$token}`
    .label = expected expression

parse_frontmatter_extra_characters_after_close = extra characters after frontmatter close are not allowed
parse_frontmatter_invalid_close_preceding_whitespace = invalid preceding whitespace for frontmatter close
    .note = frontmatter close should not be preceded by whitespace
parse_frontmatter_invalid_infostring = invalid infostring for frontmatter
    .note = frontmatter infostrings must be a single identifier immediately following the opening
parse_frontmatter_invalid_opening_preceding_whitespace = invalid preceding whitespace for frontmatter opening
    .note = frontmatter opening should not be preceded by whitespace
parse_frontmatter_length_mismatch = frontmatter close does not match the opening
    .label_opening = the opening here has {$len_opening} dashes...
    .label_close = ...while the close has {$len_close} dashes
parse_frontmatter_unclosed = unclosed frontmatter
    .note = frontmatter opening here was not closed

parse_function_body_equals_expr = function body cannot be `= expression;`
    .suggestion = surround the expression with `{"{"}` and `{"}"}` instead of `=` and `;`

parse_generic_args_in_pat_require_turbofish_syntax = generic args in patterns require the turbofish syntax

parse_generic_parameters_without_angle_brackets = generic parameters without surrounding angle brackets
    .suggestion = surround the type parameters with angle brackets

parse_generics_in_path = unexpected generic arguments in path

parse_help_set_edition_cargo = set `edition = "{$edition}"` in `Cargo.toml`
parse_help_set_edition_standalone = pass `--edition {$edition}` to `rustc`
parse_if_expression_missing_condition = missing condition for `if` expression
    .condition_label = expected condition here
    .block_label = if this block is the condition of the `if` expression, then it must be followed by another block

parse_if_expression_missing_then_block = this `if` expression is missing a block after the condition
    .add_then_block = add a block here
    .condition_possibly_unfinished = this binary operation is possibly unfinished

parse_in_in_typo =
    expected iterable, found keyword `in`
    .suggestion = remove the duplicated `in`

parse_inappropriate_default = {$article} {$descr} cannot be `default`
    .label = `default` because of this
    .note = only associated `fn`, `const`, and `type` items can be `default`

parse_inclusive_range_extra_equals = unexpected `=` after inclusive range
    .suggestion_remove_eq = use `..=` instead
    .note = inclusive ranges end with a single equals sign (`..=`)

parse_inclusive_range_match_arrow = unexpected `>` after inclusive range
    .label = this is parsed as an inclusive range `..=`
    .suggestion = add a space between the pattern and `=>`

parse_inclusive_range_no_end = inclusive range with no end
    .suggestion_open_range = use `..` instead
    .note = inclusive ranges must be bounded at the end (`..=b` or `a..=b`)

parse_incorrect_parens_trait_bounds = incorrect parentheses around trait bounds
parse_incorrect_parens_trait_bounds_sugg = fix the parentheses

parse_incorrect_semicolon =
    expected item, found `;`
    .suggestion = remove this semicolon
    .help = {$name} declarations are not followed by a semicolon

parse_incorrect_type_on_self = type not allowed for shorthand `self` parameter
    .suggestion = move the modifiers on `self` to the type

parse_incorrect_use_of_await = incorrect use of `await`
    .parentheses_suggestion = `await` is not a method call, remove the parentheses

parse_incorrect_use_of_await_postfix_suggestion = `await` is a postfix operation

parse_incorrect_use_of_use = incorrect use of `use`
    .parentheses_suggestion = `use` is not a method call, try removing the parentheses

parse_incorrect_visibility_restriction = incorrect visibility restriction
    .help = some possible visibility restrictions are:
            `pub(crate)`: visible only on the current crate
            `pub(super)`: visible only in the current module's parent
            `pub(in path::to::module)`: visible only on the specified path
    .suggestion = make this visible only to module `{$inner_str}` with `in`

parse_inner_attr_explanation = inner attributes, like `#![no_std]`, annotate the item enclosing them, and are usually found at the beginning of source files
parse_inner_attr_not_permitted = an inner attribute is not permitted in this context
    .label_does_not_annotate_this = {parse_label_inner_attr_does_not_annotate_this}
    .sugg_change_inner_to_outer = {parse_sugg_change_inner_attr_to_outer}

parse_inner_attr_not_permitted_after_outer_attr = an inner attribute is not permitted following an outer attribute
    .label_attr = not permitted following an outer attribute
    .label_prev_attr = previous outer attribute
    .label_does_not_annotate_this = {parse_label_inner_attr_does_not_annotate_this}
    .sugg_change_inner_to_outer = {parse_sugg_change_inner_attr_to_outer}

parse_inner_attr_not_permitted_after_outer_doc_comment = an inner attribute is not permitted following an outer doc comment
    .label_attr = not permitted following an outer doc comment
    .label_prev_doc_comment = previous doc comment
    .label_does_not_annotate_this = {parse_label_inner_attr_does_not_annotate_this}
    .sugg_change_inner_to_outer = {parse_sugg_change_inner_attr_to_outer}

parse_inner_doc_comment_not_permitted = expected outer doc comment
    .note = inner doc comments like this (starting with `//!` or `/*!`) can only appear before items
    .suggestion = you might have meant to write a regular comment
    .label_does_not_annotate_this = the inner doc comment doesn't annotate this {$item}
    .sugg_change_inner_to_outer = to annotate the {$item}, change the doc comment from inner to outer style

parse_invalid_attr_unsafe = `{$name}` is not an unsafe attribute
    .label = this is not an unsafe attribute
    .suggestion = remove the `unsafe(...)`
    .note = extraneous unsafe is not allowed in attributes

parse_invalid_block_macro_segment = cannot use a `block` macro fragment here
    .label = the `block` fragment is within this context
    .suggestion = wrap this in another block

parse_invalid_char_in_escape = {parse_invalid_char_in_escape_msg}: `{$ch}`
    .label = {parse_invalid_char_in_escape_msg}

parse_invalid_char_in_escape_msg = invalid character in {$is_hex ->
    [true] numeric character
    *[false] unicode
    } escape


parse_invalid_comparison_operator = invalid comparison operator `{$invalid}`
    .use_instead = `{$invalid}` is not a valid comparison operator, use `{$correct}`
    .spaceship_operator_invalid = `<=>` is not a valid comparison operator, use `std::cmp::Ordering`

parse_invalid_curly_in_let_else = right curly brace `{"}"}` before `else` in a `let...else` statement not allowed
parse_invalid_digit_literal = invalid digit for a base {$base} literal

parse_invalid_dyn_keyword = invalid `dyn` keyword
    .help = `dyn` is only needed at the start of a trait `+`-separated list
    .suggestion = remove this keyword

parse_invalid_expression_in_let_else = a `{$operator}` expression cannot be directly assigned in `let...else`
parse_invalid_identifier_with_leading_number = identifiers cannot start with a number

parse_invalid_label =
    invalid label name `{$name}`

parse_invalid_literal_suffix_on_tuple_index = suffixes on a tuple index are invalid
    .label = invalid suffix `{$suffix}`
    .tuple_exception_line_1 = `{$suffix}` is *temporarily* accepted on tuple index fields as it was incorrectly accepted on stable for a few releases
    .tuple_exception_line_2 = on proc macros, you'll want to use `syn::Index::from` or `proc_macro::Literal::*_unsuffixed` for code that will desugar to tuple field access
    .tuple_exception_line_3 = see issue #60210 <https://github.com/rust-lang/rust/issues/60210> for more information

parse_invalid_logical_operator = `{$incorrect}` is not a logical operator
    .note = unlike in e.g., Python and PHP, `&&` and `||` are used for logical operators
    .use_amp_amp_for_conjunction = use `&&` to perform logical conjunction
    .use_pipe_pipe_for_disjunction = use `||` to perform logical disjunction

parse_invalid_meta_item = expected unsuffixed literal, found {$descr}
    .quote_ident_sugg = surround the identifier with quotation marks to make it into a string literal

parse_invalid_offset_of = offset_of expects dot-separated field and variant names

parse_invalid_path_sep_in_fn_definition = invalid path separator in function definition
    .suggestion = remove invalid path separator

parse_invalid_unicode_escape = invalid unicode character escape
    .label = invalid escape
    .help = unicode escape must {$surrogate ->
    [true] not be a surrogate
    *[false] be at most 10FFFF
    }

parse_invalid_variable_declaration =
    invalid variable declaration

parse_keyword_lifetime =
    lifetimes cannot use keyword names

parse_kw_bad_case = keyword `{$kw}` is written in the wrong case
    .suggestion = write it in the correct case

parse_label_inner_attr_does_not_annotate_this = the inner attribute doesn't annotate this {$item}
parse_label_unexpected_token = unexpected token

parse_label_while_parsing_or_pattern_here = while parsing this or-pattern starting here

parse_labeled_loop_in_break = parentheses are required around this expression to avoid confusion with a labeled break expression

parse_leading_plus_not_supported = leading `+` is not supported
    .label = unexpected `+`
    .suggestion_remove_plus = try removing the `+`

parse_leading_underscore_unicode_escape = {parse_leading_underscore_unicode_escape_label}: `_`
parse_leading_underscore_unicode_escape_label = invalid start of unicode escape

parse_left_arrow_operator = unexpected token: `<-`
    .suggestion = if you meant to write a comparison against a negative value, add a space in between `<` and `-`

parse_lifetime_after_mut = lifetime must precede `mut`
    .suggestion = place the lifetime before `mut`

parse_lifetime_in_borrow_expression = borrow expressions cannot be annotated with lifetimes
    .suggestion = remove the lifetime annotation
    .label = annotated with lifetime here

parse_lifetime_in_eq_constraint = lifetimes are not permitted in this context
    .label = lifetime is not allowed here
    .context_label = this introduces an associated item binding
    .help = if you meant to specify a trait object, write `dyn /* Trait */ + {$lifetime}`
    .colon_sugg = you might have meant to write a bound here

parse_lone_slash = invalid trailing slash in literal
    .label = {parse_lone_slash}

parse_loop_else = `{$loop_kind}...else` loops are not supported
    .note = consider moving this `else` clause to a separate `if` statement and use a `bool` variable to control if it should run
    .loop_keyword = `else` is attached to this loop

parse_macro_expands_to_adt_field = macros cannot expand to {$adt_ty} fields

parse_macro_expands_to_enum_variant = macros cannot expand to enum variants

parse_macro_invocation_visibility = can't qualify macro invocation with `pub`
    .suggestion = remove the visibility
    .help = try adjusting the macro to put `{$vis}` inside the invocation

parse_macro_invocation_with_qualified_path = macros cannot use qualified paths

parse_macro_name_remove_bang = macro names aren't followed by a `!`
    .suggestion = remove the `!`

parse_macro_rules_missing_bang = expected `!` after `macro_rules`
    .suggestion = add a `!`

parse_macro_rules_visibility = can't qualify macro_rules invocation with `{$vis}`
    .suggestion = try exporting the macro

parse_malformed_cfg_attr = malformed `cfg_attr` attribute input
    .suggestion = missing condition and attribute
    .note = for more information, visit <https://doc.rust-lang.org/reference/conditional-compilation.html#the-cfg_attr-attribute>

parse_malformed_loop_label = malformed loop label
    .suggestion = use the correct loop label format

parse_match_arm_body_without_braces = `match` arm body without braces
    .label_statements = {$num_statements ->
            [one] this statement is not surrounded by a body
           *[other] these statements are not surrounded by a body
        }
    .label_arrow = while parsing the `match` arm starting here
    .suggestion_add_braces = surround the {$num_statements ->
            [one] statement
           *[other] statements
        } with a body
    .suggestion_use_comma_not_semicolon = replace `;` with `,` to end a `match` arm expression

parse_maybe_comparison = you might have meant to compare for equality

parse_maybe_fn_typo_with_impl = you might have meant to write `impl` instead of `fn`
    .suggestion = replace `fn` with `impl` here

parse_maybe_missing_let = you might have meant to continue the let-chain

parse_maybe_recover_from_bad_qpath_stage_2 =
    missing angle brackets in associated item path
    .suggestion = types that don't start with an identifier need to be surrounded with angle brackets in qualified paths

parse_maybe_recover_from_bad_type_plus =
    expected a path on the left-hand side of `+`

parse_maybe_report_ambiguous_plus =
    ambiguous `+` in a type
    .suggestion = use parentheses to disambiguate

parse_meta_bad_delim = wrong meta list delimiters
parse_meta_bad_delim_suggestion = the delimiters should be `(` and `)`

parse_mismatched_closing_delimiter = mismatched closing delimiter: `{$delimiter}`
    .label_unmatched = mismatched closing delimiter
    .label_opening_candidate = closing delimiter possibly meant for this
    .label_unclosed = unclosed delimiter

parse_misplaced_return_type = place the return type after the function parameters

parse_missing_comma_after_match_arm = expected `,` following `match` arm
    .suggestion = missing a comma here to end this `match` arm

parse_missing_const_type = missing type for `{$kind}` item
    .suggestion = provide a type for the item

parse_missing_enum_for_enum_definition = missing `enum` for enum definition
    .suggestion = add `enum` here to parse `{$ident}` as an enum

parse_missing_enum_or_struct_for_item_definition = missing `enum` or `struct` for enum or struct definition

parse_missing_expression_in_for_loop = missing expression to iterate on in `for` loop
    .suggestion = try adding an expression to the `for` loop

parse_missing_fn_for_function_definition = missing `fn` for function definition
    .suggestion = add `fn` here to parse `{$ident}` as a function

parse_missing_fn_for_method_definition = missing `fn` for method definition
    .suggestion = add `fn` here to parse `{$ident}` as a method

parse_missing_fn_or_struct_for_item_definition = missing `fn` or `struct` for function or struct definition
    .suggestion = if you meant to call a macro, try
    .help = if you meant to call a macro, remove the `pub` and add a trailing `!` after the identifier

parse_missing_fn_params = missing parameters for function definition
    .suggestion = add a parameter list

parse_missing_for_in_trait_impl = missing `for` in a trait impl
    .suggestion = add `for` here

parse_missing_in_in_for_loop = missing `in` in `for` loop
    .use_in_not_of = try using `in` here instead
    .add_in = try adding `in` here

parse_missing_let_before_mut = missing keyword
parse_missing_plus_in_bounds = expected `+` between lifetime and {$sym}
    .suggestion = add `+`

parse_missing_semicolon_before_array = expected `;`, found `[`
    .suggestion = consider adding `;` here

parse_missing_struct_for_struct_definition = missing `struct` for struct definition
    .suggestion = add `struct` here to parse `{$ident}` as a struct

parse_missing_trait_in_trait_impl = missing trait in a trait impl
    .suggestion_add_trait = add a trait here
    .suggestion_remove_for = for an inherent impl, drop this `for`

parse_misspelled_kw = {$is_incorrect_case ->
                    [true] write keyword `{$similar_kw}` in lowercase
                    *[false] there is a keyword `{$similar_kw}` with a similar name
}

parse_modifier_lifetime = `{$modifier}` may only modify trait bounds, not lifetime bounds
    .suggestion = remove the `{$modifier}`

parse_modifiers_and_polarity = `{$modifiers_concatenated}` trait not allowed with `{$polarity}` trait polarity modifier
    .label = there is not a well-defined meaning for a `{$modifiers_concatenated} {$polarity}` trait

parse_more_than_one_char = character literal may only contain one codepoint
    .followed_by = this `{$chr}` is followed by the combining {$len ->
        [one] mark
        *[other] marks
        } `{$escaped_marks}`
    .non_printing = there are non-printing characters, the full sequence is `{$escaped}`
    .consider_normalized = consider using the normalized form `{$ch}` of this character
    .remove_non = consider removing the non-printing characters
    .use_double_quotes = if you meant to write a {$is_byte ->
        [true] byte string
        *[false] string
        } literal, use double quotes

parse_multiple_skipped_lines = multiple lines skipped by escaped newline
    .label = skipping everything up to and including this point

parse_multiple_where_clauses = cannot define duplicate `where` clauses on an item
    .label = previous `where` clause starts here
    .suggestion = consider joining the two `where` clauses into one

parse_mut_on_nested_ident_pattern = `mut` must be attached to each individual binding
    .suggestion = add `mut` to each binding
parse_mut_on_non_ident_pattern = `mut` must be followed by a named binding
    .suggestion = remove the `mut` prefix

parse_need_plus_after_trait_object_lifetime = lifetimes must be followed by `+` to form a trait object type
    .suggestion = consider adding a trait bound after the potential lifetime bound

parse_nested_adt = `{$kw_str}` definition cannot be nested inside `{$keyword}`
    .suggestion = consider creating a new `{$kw_str}` definition instead of nesting

parse_nested_c_variadic_type = C-variadic type `...` may not be nested inside another type

parse_no_brace_unicode_escape = incorrect unicode escape sequence
    .label = {parse_no_brace_unicode_escape}
    .use_braces = format of unicode escape sequences uses braces
    .format_of_unicode = format of unicode escape sequences is `\u{"{...}"}`

parse_no_digits_literal = no valid digits found for number

parse_non_string_abi_literal = non-string ABI literal
    .suggestion = specify the ABI with a string literal

parse_nonterminal_expected_ident = expected ident, found `{$token}`
parse_nonterminal_expected_item_keyword = expected an item keyword
parse_nonterminal_expected_lifetime = expected a lifetime, found `{$token}`

parse_nonterminal_expected_statement = expected a statement

parse_note_edition_guide = for more on editions, read https://doc.rust-lang.org/edition-guide

parse_note_mut_pattern_usage = `mut` may be followed by `variable` and `variable @ pattern`

parse_note_pattern_alternatives_use_single_vert = alternatives in or-patterns are separated with `|`, not `||`

parse_nul_in_c_str = null characters in C string literals are not supported

parse_or_in_let_chain = `||` operators are not supported in let chain conditions

parse_or_pattern_not_allowed_in_fn_parameters = function parameters require top-level or-patterns in parentheses
parse_or_pattern_not_allowed_in_let_binding = `let` bindings require top-level or-patterns in parentheses
parse_out_of_range_hex_escape = out of range hex escape
    .label = must be a character in the range [\x00-\x7f]

parse_outer_attr_explanation = outer attributes, like `#[test]`, annotate the item following them

parse_outer_attribute_not_allowed_on_if_else = outer attributes are not allowed on `if` and `else` branches
    .branch_label = the attributes are attached to this branch
    .ctx_label = the branch belongs to this `{$ctx}`
    .suggestion = remove the attributes

parse_overlong_unicode_escape = overlong unicode escape
    .label = must have at most 6 hex digits

parse_parentheses_with_struct_fields = invalid `struct` delimiters or `fn` call arguments
    .suggestion_braces_for_struct = if `{$type}` is a struct, use braces as delimiters
    .suggestion_no_fields_for_fn = if `{$type}` is a function, use the arguments directly

parse_parenthesized_lifetime = parenthesized lifetime bounds are not supported
parse_parenthesized_lifetime_suggestion = remove the parentheses

parse_path_double_colon = path separator must be a double colon
    .suggestion = use a double colon instead


parse_path_found_attribute_in_params = `Trait(...)` syntax does not support attributes in parameters
    .suggestion = remove the attributes

parse_path_found_c_variadic_params = `Trait(...)` syntax does not support c_variadic parameters
    .suggestion = remove the `...`

parse_path_found_named_params = `Trait(...)` syntax does not support named parameters
    .suggestion = remove the parameter name

parse_pattern_method_param_without_body = patterns aren't allowed in methods without bodies
    .suggestion = give this argument a name or use an underscore to ignore it

parse_pattern_on_wrong_side_of_at = pattern on wrong side of `@`
    .label_pattern = pattern on the left, should be on the right
    .label_binding = binding on the right, should be on the left
    .suggestion = switch the order

parse_question_mark_in_type = invalid `?` in type
    .label = `?` is only allowed on expressions, not types
    .suggestion = if you meant to express that the type might not contain a value, use the `Option` wrapper type

parse_recover_import_as_use = expected item, found {$token_name}
    .suggestion = items are imported using the `use` keyword

parse_remove_let = expected pattern, found `let`
    .suggestion = remove the unnecessary `let` keyword

parse_repeated_mut_in_pattern = `mut` on a binding may not be repeated
    .suggestion = remove the additional `mut`s

parse_require_colon_after_labeled_expression = labeled expression must be followed by `:`
    .note = labels are used before loops and blocks, allowing e.g., `break 'label` to them
    .label = the label
    .suggestion = add `:` after the label

parse_reserved_multihash = reserved multi-hash token is forbidden
    .note = sequences of two or more # are reserved for future use since Rust 2024
    .suggestion_whitespace = consider inserting whitespace here

parse_reserved_string = invalid string literal
    .note = unprefixed guarded string literals are reserved for future use since Rust 2024
    .suggestion_whitespace = consider inserting whitespace here

parse_return_types_use_thin_arrow = return types are denoted using `->`
    .suggestion = use `->` instead

parse_self_argument_pointer = cannot pass `self` by raw pointer
    .label = cannot pass `self` by raw pointer

parse_self_param_not_first = unexpected `self` parameter in function
    .label = must be the first parameter of an associated function

parse_shift_interpreted_as_generic =
    `<<` is interpreted as a start of generic arguments for `{$type}`, not a shift
    .label_args = interpreted as generic arguments
    .label_comparison = not interpreted as shift
    .suggestion = try shifting the cast value

parse_single_colon_import_path = expected `::`, found `:`
    .suggestion = use double colon
    .note = import paths are delimited using `::`

parse_static_with_generics = static items may not have generic parameters

parse_struct_literal_body_without_path =
    struct literal body without path
    .suggestion = you might have forgotten to add the struct literal inside the block

parse_struct_literal_not_allowed_here = struct literals are not allowed here
    .suggestion = surround the struct literal with parentheses

parse_suffixed_literal_in_attribute = suffixed literals are not allowed in attributes
    .help = instead of using a suffixed literal (`1u8`, `1.0f32`, etc.), use an unsuffixed version (`1`, `1.0`, etc.)

parse_sugg_add_let_for_stmt = you might have meant to introduce a new binding

parse_sugg_add_semi = add `;` here
parse_sugg_change_inner_attr_to_outer = to annotate the {$item}, change the attribute from inner to outer style

parse_sugg_change_this_to_semi = change this to `;`
parse_sugg_escape_identifier = escape `{$ident_name}` to use it as an identifier

parse_sugg_remove_comma = remove this comma
parse_sugg_remove_leading_vert_in_pattern = remove the `|`
parse_sugg_turbofish_syntax = use `::<...>` instead of `<...>` to specify lifetime, type, or const arguments

parse_sugg_wrap_expression_in_parentheses = wrap the expression in parentheses

parse_sugg_wrap_macro_in_parentheses = use parentheses instead of braces for this macro

parse_sugg_wrap_pattern_in_parens = wrap the pattern in parentheses

parse_switch_mut_let_order =
    switch the order of `mut` and `let`

parse_switch_ref_box_order = switch the order of `ref` and `box`
    .suggestion = swap them

parse_ternary_operator = Rust has no ternary operator

parse_tilde_is_not_unary_operator = `~` cannot be used as a unary operator
    .suggestion = use `!` to perform bitwise not

parse_too_many_hashes = too many `#` symbols: raw strings may be delimited by up to 255 `#` symbols, but found {$num}

parse_too_short_hex_escape = numeric character escape is too short

parse_trailing_vert_not_allowed = a trailing `|` is not allowed in an or-pattern
    .suggestion = remove the `{$token}`

parse_trait_alias_cannot_be_auto = trait aliases cannot be `auto`
parse_trait_alias_cannot_be_unsafe = trait aliases cannot be `unsafe`

parse_transpose_dyn_or_impl = `for<...>` expected after `{$kw}`, not before
    .suggestion = move `{$kw}` before the `for<...>`

parse_unclosed_unicode_escape = unterminated unicode escape
    .label = missing a closing `{"}"}`
    .terminate = terminate the unicode escape

parse_underscore_literal_suffix = underscore literal suffix is not allowed

parse_unexpected_const_in_generic_param = expected lifetime, type, or constant, found keyword `const`
    .suggestion = the `const` keyword is only needed in the definition of the type

parse_unexpected_const_param_declaration = unexpected `const` parameter declaration
    .label = expected a `const` expression, not a parameter declaration
    .suggestion = `const` parameters must be declared for the `impl`

parse_unexpected_default_value_for_lifetime_in_generic_parameters = unexpected default lifetime parameter
    .label = lifetime parameters cannot have default values

parse_unexpected_expr_in_pat =
    expected {$is_bound ->
        [true] a pattern range bound
       *[false] a pattern
    }, found an expression

    .label = not a pattern
    .note = arbitrary expressions are not allowed in patterns: <https://doc.rust-lang.org/book/ch19-00-patterns.html>

parse_unexpected_expr_in_pat_const_sugg = consider extracting the expression into a `const`

parse_unexpected_expr_in_pat_create_guard_sugg = consider moving the expression to a match arm guard

parse_unexpected_expr_in_pat_update_guard_sugg = consider moving the expression to the match arm guard

parse_unexpected_if_with_if = unexpected `if` in the condition expression
    .suggestion = remove the `if`

parse_unexpected_lifetime_in_pattern = unexpected lifetime `{$symbol}` in pattern
    .suggestion = remove the lifetime

parse_unexpected_paren_in_range_pat = range pattern bounds cannot have parentheses
parse_unexpected_paren_in_range_pat_sugg = remove these parentheses

parse_unexpected_parentheses_in_for_head = unexpected parentheses surrounding `for` loop head
    .suggestion = remove parentheses in `for` loop

parse_unexpected_parentheses_in_match_arm_pattern = unexpected parentheses surrounding `match` arm pattern
    .suggestion = remove parentheses surrounding the pattern

parse_unexpected_self_in_generic_parameters = unexpected keyword `Self` in generic parameters
    .note = you cannot use `Self` as a generic parameter because it is reserved for associated items

parse_unexpected_token_after_dot = unexpected token: {$actual}

parse_unexpected_token_after_label = expected `while`, `for`, `loop` or `{"{"}` after a label
    .suggestion_remove_label = consider removing the label
    .suggestion_enclose_in_block = consider enclosing expression in a block

parse_unexpected_token_after_not = unexpected {$negated_desc} after identifier
parse_unexpected_token_after_not_bitwise = use `!` to perform bitwise not
parse_unexpected_token_after_not_default = use `!` to perform logical negation or bitwise not

parse_unexpected_token_after_not_logical = use `!` to perform logical negation
parse_unexpected_token_after_struct_name = expected `where`, `{"{"}`, `(`, or `;` after struct name
parse_unexpected_token_after_struct_name_found_doc_comment = expected `where`, `{"{"}`, `(`, or `;` after struct name, found doc comment `{$token}`
parse_unexpected_token_after_struct_name_found_keyword = expected `where`, `{"{"}`, `(`, or `;` after struct name, found keyword `{$token}`
# This one deliberately doesn't print a token.
parse_unexpected_token_after_struct_name_found_metavar = expected `where`, `{"{"}`, `(`, or `;` after struct name, found metavar
parse_unexpected_token_after_struct_name_found_other = expected `where`, `{"{"}`, `(`, or `;` after struct name, found `{$token}`

parse_unexpected_token_after_struct_name_found_reserved_identifier = expected `where`, `{"{"}`, `(`, or `;` after struct name, found reserved identifier `{$token}`
parse_unexpected_token_after_struct_name_found_reserved_keyword = expected `where`, `{"{"}`, `(`, or `;` after struct name, found reserved keyword `{$token}`
parse_unexpected_vert_vert_before_function_parameter = unexpected `||` before function parameter
    .suggestion = remove the `||`

parse_unexpected_vert_vert_in_pattern = unexpected token `||` in pattern
    .suggestion = use a single `|` to separate multiple alternative patterns

parse_unicode_escape_in_byte = unicode escape in byte string
    .label = {parse_unicode_escape_in_byte}
    .help = unicode escape sequences cannot be used as a byte or in a byte string

parse_unknown_builtin_construct = unknown `builtin #` construct `{$name}`

parse_unknown_prefix = prefix `{$prefix}` is unknown
    .label = unknown prefix
    .note =  prefixed identifiers and literals are reserved since Rust 2021
    .suggestion_br = use `br` for a raw byte string
    .suggestion_cr = use `cr` for a raw C-string
    .suggestion_str = if you meant to write a string literal, use double quotes
    .suggestion_whitespace = consider inserting whitespace here

parse_unknown_start_of_token = unknown start of token: {$escaped}
    .sugg_quotes = Unicode characters '“' (Left Double Quotation Mark) and '”' (Right Double Quotation Mark) look like '{$ascii_str}' ({$ascii_name}), but are not
    .sugg_other = Unicode character '{$ch}' ({$u_name}) looks like '{$ascii_str}' ({$ascii_name}), but it is not
    .help_null = source files must contain UTF-8 encoded text, unexpected null bytes might occur when a different encoding is used
    .note_repeats = character appears {$repeats ->
        [one] once more
        *[other] {$repeats} more times
    }

parse_unmatched_angle = unmatched angle {$plural ->
    [true] brackets
    *[false] bracket
    }
    .suggestion = remove extra angle {$plural ->
    [true] brackets
    *[false] bracket
    }

parse_unmatched_angle_brackets = {$num_extra_brackets ->
        [one] unmatched angle bracket
       *[other] unmatched angle brackets
    }
    .suggestion = {$num_extra_brackets ->
            [one] remove extra angle bracket
           *[other] remove extra angle brackets
        }

parse_unsafe_attr_outside_unsafe = unsafe attribute used without unsafe
    .label = usage of unsafe attribute
parse_unsafe_attr_outside_unsafe_suggestion = wrap the attribute in `unsafe(...)`


parse_unskipped_whitespace = whitespace symbol '{$ch}' is not skipped
    .label = {parse_unskipped_whitespace}

parse_use_empty_block_not_semi = expected { "`{}`" }, found `;`
    .suggestion = try using { "`{}`" } instead

parse_use_eq_instead = unexpected `==`
    .suggestion = try using `=` instead

parse_use_if_else = use an `if-else` expression instead

parse_use_let_not_auto = write `let` instead of `auto` to introduce a new variable
parse_use_let_not_var = write `let` instead of `var` to introduce a new variable

parse_visibility_not_followed_by_item = visibility `{$vis}` is not followed by an item
    .label = the visibility
    .help = you likely meant to define an item, e.g., `{$vis} fn foo() {"{}"}`

parse_where_clause_before_const_body = where clauses are not allowed before const item bodies
    .label = unexpected where clause
    .name_label = while parsing this const item
    .body_label = the item body
    .suggestion = move the body before the where clause

parse_where_clause_before_tuple_struct_body = where clauses are not allowed before tuple struct bodies
    .label = unexpected where clause
    .name_label = while parsing this tuple struct
    .body_label = the struct body
    .suggestion = move the body before the where clause

parse_where_generics = generic parameters on `where` clauses are reserved for future use
    .label = currently unsupported

parse_zero_chars = empty character literal
    .label = {parse_zero_chars}
