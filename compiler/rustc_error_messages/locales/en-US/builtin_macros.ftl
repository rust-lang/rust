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
