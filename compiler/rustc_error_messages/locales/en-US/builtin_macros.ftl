builtin_macros_asm_args_after_clobber_abi =
    arguments are not allowed after clobber_abi
    .label = argument
    .abi_label = clobber_abi

builtin_macros_asm_args_after_options =
    arguments are not allowed after options
    .label = argument
    .previous_options_label = previous options

builtin_macros_asm_args_named_after_explicit_register =
    named arguments cannot follow explicit register arguments
    .label = named argument
    .register_label = explicit register argument

builtin_macros_asm_args_positional_after_named_or_explicit_register =
    positional arguments cannot follow named arguments or explicit register arguments
    .label = positional argument
    .named_label = named argument
    .register_label = explicit register argument

builtin_macros_asm_cannot_be_used_with = `{$left}` cannot be used with `{$right}`

builtin_macros_asm_clobber_abi_after_options =
    clobber_abi is not allowed after options
    .options_label = options

builtin_macros_asm_clobber_abi_needs_an_abi =
    at least one abi must be provided as an argument to `clobber_abi`

builtin_macros_asm_clobber_abi_needs_explicit_registers =
    asm with `clobber_abi` must specify explicit registers for outputs
    .label = generic outputs
    .abi_label = clobber_abi

builtin_macros_asm_duplicate_argument =
    duplicate argument named `{$name}`
    .label = duplicate argument
    .previously_label = previously here

builtin_macros_asm_duplicate_option =
    the `{$symbol}` option was already provided
    .label = this option was already provided
    .suggestion = remove this option

builtin_macros_asm_expected_operand_options_or_template_string =
    expected operand, options, or additional template string
    .label = expected operand, options, or additional template string

builtin_macros_asm_expected_operand_clobber_abi_options_or_template_string =
    expected operand, clobber_abi, options, or additional template string
    .label = expected operand, clobber_abi, options, or additional template string

builtin_macros_asm_expected_path_arg_to_sym = expected a path for argument to `sym`

builtin_macros_asm_expected_register_class_or_explicit_register =
    expected register class or explicit register

builtin_macros_asm_expected_string_literal =
    expected string literal
    .label = not a string literal

builtin_macros_asm_expected_token_comma =
    expected token: `,`
    .label = expected `,`

builtin_macros_asm_explicit_register_arg_with_name = explicit register arguments cannot have names

builtin_macros_asm_no_argument_named = there is no argument named `{$name}`

builtin_macros_asm_option_must_be_combined_with_either =
    the `{$option}` option must be combined with either `{$left}` or `{$right}`

builtin_macros_asm_option_noreturn_with_outputs =
    asm outputs are not allowed with the `noreturn` option

builtin_macros_asm_option_pure_needs_one_output =
    asm with the `pure` option must have at least one output

builtin_macros_asm_options_mutually_exclusive =
    the `{$left}` and `${right}` options are mutually exclusive

builtin_macros_asm_requires_template_string_arg = requires at least a template string argument

builtin_macros_asm_template_modifier_single_char = asm template modifier must be a single character

builtin_macros_asm_underscore_for_input_operands = _ cannot be used for input operands

builtin_macros_requires_cfg_pattern =
    macro requires a cfg-pattern as an argument
    .label = cfg-pattern required

builtin_macros_expected_one_cfg_pattern = expected 1 cfg-pattern

builtin_macros_boolean_expression_required =
    macro requires a boolean expression as an argument
    .label = boolean expression required

builtin_macros_unexpected_string_literal = unexpected string literal
    .suggestion = try adding a comma

builtin_macros_argument_expression_required =
    macro requires an expression as an argument
    .suggestion = try removing semicolon

builtin_macros_not_specified =
    `cfg_accessible` path is not specified

builtin_macros_multiple_paths_specified =
    multiple `cfg_accessible` paths are specified

builtin_macros_unallowed_literal_path =
    `cfg_accessible` path cannot be a literal

builtin_macros_unaccepted_arguments =
    `cfg_accessible` path cannot accept arguments

builtin_macros_nondeterministic_access =
    cannot determine whether the path is accessible or not

builtin_macros_compile_error = {$msg}

builtin_macros_byte_string_literal_concatenate =
    cannot concatenate a byte string literal

builtin_macros_missing_literal = expected a literal
    .note = only literals (like `\"foo\"`, `42` and `3.14`) can be passed to `concat!()`

builtin_macros_character_literals_concatenate = cannot concatenate character literals

builtin_macros_string_literals_concatenate = cannot concatenate string literals

builtin_macros_use_byte_character = try using a byte character

builtin_macros_use_byte_string = try using a byte string

builtin_macros_float_literals_concatenate = cannot concatenate float literals

builtin_macros_boolean_literals_concatenate = cannot concatenate boolean literals

builtin_macros_wrap_number_in_array = try wrapping the number in an array

builtin_macros_numeric_literals_concatenate = cannot concatenate numeric literals

builtin_macros_out_of_bound_numeric_literal = numeric literal is out of bounds

builtin_macros_invalid_numeric_literal = numeric literal is not a `u8`

builtin_macros_doubly_nested_array_concatenate = cannot concatenate doubly nested array
    .note = byte strings are treated as arrays of bytes
    .help = try flattening the array

builtin_macros_invalid_repeat_count = repeat count is not a positive number

builtin_macros_byte_literal_expected = expected a byte literal
    .note = only byte literals (like `b\"foo\"`, `b's'`, and `[3, 4, 5]`) can be passed to `concat_bytes!()`

builtin_macros_missing_arguments = concat_idents! takes 1 or more arguments

builtin_macros_comma_expected = concat_idents! expecting comma

builtin_macros_ident_args_required = concat_idents! requires ident args

builtin_macros_not_applicable_derive =
    `derive` may only be applied to `struct`s, `enum`s and `union`s
    .label = not applicable here
    .item_label = not a `struct`, `enum` or `union`

builtin_macros_trait_path_expected =
    expected path to a trait, found literal
    .label = not a trait
    .help = {$help_msg}

builtin_macros_path_rejected = {$title}
    .suggestion = {$action}

builtin_macros_no_default = no default declared
    .help = make a unit variant default by placing `#[default]` above it

builtin_macros_variant_suggestion = make `{$ident}` default

builtin_macros_multiple_declared_defaults_first = = first default

builtin_macros_multiple_declared_defaults_additional = additional default

builtin_macros_multiple_declared_defaults = multiple declared defaults
    .note = only one variant can be default

builtin_macros_default_not_allowed =
    the `#[default]` attribute may only be used on unit enum variants
    .help = consider a manual implementation of `Default`

builtin_macros_default_non_exhaustive = default variant must be exhaustive
    .help = consider a manual implementation of `Default`

builtin_macros_default_non_exhaustive_instruction = declared `#[non_exhaustive]` here

builtin_macros_multiple_default_attributes_first = `#[default]` used here

builtin_macros_multiple_default_attributes_rest = `#[default]` used again here

builtin_macros_multiple_default_attributes_suggestion_text = {$suggestion_text}

builtin_macros_multiple_default_attributes = multiple `#[default]` attributes
    .note = only one `#[default]` attribute is needed

builtin_macros_default_not_accept_value = `#[default]` attribute does not accept a value
    .suggestion = try using `#[default]`

builtin_macros_unallowed_derive = `derive` cannot be used on items with type macros

builtin_macros_cannot_be_derived_unions = this trait cannot be derived for unions

builtin_macros_empty_argument = env! takes 1 or 2 arguments

builtin_macros_env_expand_error = {$msg}

builtin_macros_format_string_argument_required = requires at least a format string argument

builtin_macros_duplicated_argument = duplicate argument named `{$ident}`
    .label = duplicate argument

builtin_macros_duplicated_argument_prev = previously here

builtin_macros_invalid_positional_arguments = positional arguments cannot follow named arguments
    .label = positional arguments must be before named arguments

builtin_macros_invalid_positional_arguments_names = named argument
