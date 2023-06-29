builtin_macros_alloc_error_must_be_fn = alloc_error_handler must be a function
builtin_macros_alloc_must_statics = allocators must be statics

builtin_macros_asm_clobber_abi = clobber_abi
builtin_macros_asm_clobber_no_reg = asm with `clobber_abi` must specify explicit registers for outputs
builtin_macros_asm_clobber_outputs = generic outputs

builtin_macros_asm_duplicate_arg = duplicate argument named `{$name}`
    .label = previously here
    .arg = duplicate argument

builtin_macros_asm_expected_comma = expected token: `,`
    .label = expected `,`

builtin_macros_asm_expected_other = expected operand, {$is_global_asm ->
    [true] options
    *[false] clobber_abi, options
    }, or additional template string

builtin_macros_asm_explicit_register_name = explicit register arguments cannot have names

builtin_macros_asm_modifier_invalid = asm template modifier must be a single character

builtin_macros_asm_mutually_exclusive = the `{$opt1}` and `{$opt2}` options are mutually exclusive

builtin_macros_asm_noreturn = asm outputs are not allowed with the `noreturn` option

builtin_macros_asm_opt_already_provided = the `{$symbol}` option was already provided
    .label = this option was already provided
    .suggestion = remove this option

builtin_macros_asm_pos_after = positional arguments cannot follow named arguments or explicit register arguments
    .pos = positional argument
    .named = named argument
    .explicit = explicit register argument

builtin_macros_asm_pure_combine = the `pure` option must be combined with either `nomem` or `readonly`

builtin_macros_asm_pure_no_output = asm with the `pure` option must have at least one output

builtin_macros_asm_requires_template = requires at least a template string argument

builtin_macros_asm_sym_no_path = expected a path for argument to `sym`

builtin_macros_asm_underscore_input = _ cannot be used for input operands

builtin_macros_assert_missing_comma = unexpected string literal
    .suggestion = try adding a comma

builtin_macros_assert_requires_boolean = macro requires a boolean expression as an argument
    .label = boolean expression required

builtin_macros_assert_requires_expression = macro requires an expression as an argument
    .suggestion = try removing semicolon

builtin_macros_bad_derive_target = `derive` may only be applied to `struct`s, `enum`s and `union`s
    .label = not applicable here
    .label2 = not a `struct`, `enum` or `union`

builtin_macros_bench_sig = functions used as benches must have signature `fn(&mut Bencher) -> impl Termination`


builtin_macros_cannot_derive_union = this trait cannot be derived for unions

builtin_macros_cfg_accessible_has_args = `cfg_accessible` path cannot accept arguments

builtin_macros_cfg_accessible_indeterminate = cannot determine whether the path is accessible or not

builtin_macros_cfg_accessible_literal_path = `cfg_accessible` path cannot be a literal
builtin_macros_cfg_accessible_multiple_paths = multiple `cfg_accessible` paths are specified
builtin_macros_cfg_accessible_unspecified_path = `cfg_accessible` path is not specified
builtin_macros_concat_bytes_array = cannot concatenate doubly nested array
    .note = byte strings are treated as arrays of bytes
    .help = try flattening the array

builtin_macros_concat_bytes_bad_repeat = repeat count is not a positive number

builtin_macros_concat_bytes_invalid = cannot concatenate {$lit_kind} literals
    .byte_char = try using a byte character
    .byte_str = try using a byte string
    .number_array = try wrapping the number in an array

builtin_macros_concat_bytes_missing_literal = expected a byte literal
    .note = only byte literals (like `b"foo"`, `b's'` and `[3, 4, 5]`) can be passed to `concat_bytes!()`

builtin_macros_concat_bytes_non_u8 = numeric literal is not a `u8`

builtin_macros_concat_bytes_oob = numeric literal is out of bounds

builtin_macros_concat_bytestr = cannot concatenate a byte string literal
builtin_macros_concat_c_str_lit = cannot concatenate a C string literal

builtin_macros_concat_idents_ident_args = `concat_idents!()` requires ident args

builtin_macros_concat_idents_missing_args = `concat_idents!()` takes 1 or more arguments
builtin_macros_concat_idents_missing_comma = `concat_idents!()` expecting comma
builtin_macros_concat_missing_literal = expected a literal
    .note = only literals (like `"foo"`, `-42` and `3.14`) can be passed to `concat!()`

builtin_macros_default_arg = `#[default]` attribute does not accept a value
    .suggestion = try using `#[default]`

builtin_macros_derive_macro_call = `derive` cannot be used on items with type macros

builtin_macros_derive_path_args_list = traits in `#[derive(...)]` don't accept arguments
    .suggestion = remove the arguments

builtin_macros_derive_path_args_value = traits in `#[derive(...)]` don't accept values
    .suggestion = remove the value

builtin_macros_env_not_defined = environment variable `{$var}` not defined at compile time
    .cargo = Cargo sets build script variables at run time. Use `std::env::var("{$var}")` instead
    .other = use `std::env::var("{$var}")` to read the variable at run time

builtin_macros_env_takes_args = `env!()` takes 1 or 2 arguments

builtin_macros_expected_one_cfg_pattern = expected 1 cfg-pattern

builtin_macros_expected_register_class_or_explicit_register = expected register class or explicit register

builtin_macros_export_macro_rules = cannot export macro_rules! macros from a `proc-macro` crate type currently

builtin_macros_format_duplicate_arg = duplicate argument named `{$ident}`
    .label1 = previously here
    .label2 = duplicate argument

builtin_macros_format_no_arg_named = there is no argument named `{$name}`
    .note = did you intend to capture a variable `{$name}` from the surrounding scope?
    .note2 = to avoid ambiguity, `format_args!` cannot capture variables when the format string is expanded from a macro

builtin_macros_format_pos_mismatch = {$n} positional {$n ->
    [one] argument
    *[more] arguments
    } in format string, but {$desc}

builtin_macros_format_positional_after_named = positional arguments cannot follow named arguments
    .label = positional arguments must be before named arguments
    .named_args = named argument

builtin_macros_format_requires_string = requires at least a format string argument

builtin_macros_format_string_invalid = invalid format string: {$desc}
    .label = {$label1} in format string
    .note = {$note}
    .second_label = {$label}

builtin_macros_format_unknown_trait = unknown format trait `{$ty}`
    .note = the only appropriate formatting traits are:
                                            - ``, which uses the `Display` trait
                                            - `?`, which uses the `Debug` trait
                                            - `e`, which uses the `LowerExp` trait
                                            - `E`, which uses the `UpperExp` trait
                                            - `o`, which uses the `Octal` trait
                                            - `p`, which uses the `Pointer` trait
                                            - `b`, which uses the `Binary` trait
                                            - `x`, which uses the `LowerHex` trait
                                            - `X`, which uses the `UpperHex` trait
    .suggestion = use the `{$trait_name}` trait

builtin_macros_format_unused_arg = {$named ->
    [true] named argument
    *[false] argument
    } never used

builtin_macros_format_unused_args = multiple unused formatting arguments
    .label = multiple missing formatting specifiers

builtin_macros_global_asm_clobber_abi = `clobber_abi` cannot be used with `global_asm!`

builtin_macros_invalid_crate_attribute = invalid crate attribute

builtin_macros_multiple_default_attrs = multiple `#[default]` attributes
    .note = only one `#[default]` attribute is needed
    .label = `#[default]` used here
    .label_again = `#[default]` used again here
    .help = try removing {$only_one ->
    [true] this
    *[false] these
    }

builtin_macros_multiple_defaults = multiple declared defaults
    .label = first default
    .additional = additional default
    .note = only one variant can be default
    .suggestion = make `{$ident}` default

builtin_macros_no_default_variant = no default declared
    .help = make a unit variant default by placing `#[default]` above it
    .suggestion = make `{$ident}` default

builtin_macros_non_abi = at least one abi must be provided as an argument to `clobber_abi`

builtin_macros_non_exhaustive_default = default variant must be exhaustive
    .label = declared `#[non_exhaustive]` here
    .help = consider a manual implementation of `Default`

builtin_macros_non_unit_default = the `#[default]` attribute may only be used on unit enum variants
    .help = consider a manual implementation of `Default`

builtin_macros_proc_macro = `proc-macro` crate types currently cannot export any items other than functions tagged with `#[proc_macro]`, `#[proc_macro_derive]`, or `#[proc_macro_attribute]`

builtin_macros_requires_cfg_pattern =
    macro requires a cfg-pattern as an argument
    .label = cfg-pattern required

builtin_macros_should_panic = functions using `#[should_panic]` must return `()`

builtin_macros_sugg = consider using a positional formatting argument instead

builtin_macros_test_arg_non_lifetime = functions used as tests can not have any non-lifetime generic parameters

builtin_macros_test_args = functions used as tests can not have any arguments

builtin_macros_test_bad_fn = {$kind} functions cannot be used for tests
    .label = `{$kind}` because of this

builtin_macros_test_case_non_item = `#[test_case]` attribute is only allowed on items

builtin_macros_test_runner_invalid = `test_runner` argument must be a path
builtin_macros_test_runner_nargs = `#![test_runner(..)]` accepts exactly 1 argument

builtin_macros_tests_not_support = building tests with panic=abort is not supported without `-Zpanic_abort_tests`

builtin_macros_trace_macros = trace_macros! accepts only `true` or `false`

builtin_macros_unexpected_lit = expected path to a trait, found literal
    .label = not a trait
    .str_lit = try using `#[derive({$sym})]`
    .other = for example, write `#[derive(Debug)]` for `Debug`
