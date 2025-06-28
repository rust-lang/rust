builtin_macros_alloc_error_must_be_fn = alloc_error_handler must be a function
builtin_macros_alloc_must_statics = allocators must be statics

builtin_macros_asm_attribute_not_supported =
    this attribute is not supported on assembly
builtin_macros_asm_cfg =
    the `#[cfg(/* ... */)]` and `#[cfg_attr(/* ... */)]` attributes on assembly are unstable

builtin_macros_asm_clobber_abi = clobber_abi
builtin_macros_asm_clobber_no_reg = asm with `clobber_abi` must specify explicit registers for outputs
builtin_macros_asm_clobber_outputs = generic outputs

builtin_macros_asm_duplicate_arg = duplicate argument named `{$name}`
    .label = previously here
    .arg = duplicate argument

builtin_macros_asm_explicit_register_name = explicit register arguments cannot have names

builtin_macros_asm_mayunwind = asm labels are not allowed with the `may_unwind` option

builtin_macros_asm_modifier_invalid = asm template modifier must be a single character

builtin_macros_asm_mutually_exclusive = the `{$opt1}` and `{$opt2}` options are mutually exclusive

builtin_macros_asm_no_matched_argument_name = there is no argument named `{$name}`

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

builtin_macros_asm_unsupported_clobber_abi = `clobber_abi` cannot be used with `{$macro_name}!`

builtin_macros_asm_unsupported_option = the `{$symbol}` option cannot be used with `{$macro_name}!`
    .label = the `{$symbol}` option is not meaningful for global-scoped inline assembly
    .suggestion = remove this option

builtin_macros_assert_missing_comma = unexpected string literal
    .suggestion = try adding a comma

builtin_macros_assert_requires_boolean = macro requires a boolean expression as an argument
    .label = boolean expression required

builtin_macros_assert_requires_expression = macro requires an expression as an argument
    .suggestion = try removing semicolon

builtin_macros_autodiff = autodiff must be applied to function
builtin_macros_autodiff_missing_config = autodiff requires at least a name and mode
builtin_macros_autodiff_mode_activity = {$act} can not be used in {$mode} Mode
builtin_macros_autodiff_not_build = this rustc version does not support autodiff
builtin_macros_autodiff_number_activities = expected {$expected} activities, but found {$found}
builtin_macros_autodiff_ret_activity = invalid return activity {$act} in {$mode} Mode
builtin_macros_autodiff_ty_activity = {$act} can not be used for this type
builtin_macros_autodiff_unknown_activity = did not recognize Activity: `{$act}`

builtin_macros_autodiff_width = autodiff width must fit u32, but is {$width}
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

builtin_macros_coerce_pointee_requires_maybe_sized = `derive(CoercePointee)` requires `{$name}` to be marked `?Sized`

builtin_macros_coerce_pointee_requires_one_field = `CoercePointee` can only be derived on `struct`s with at least one field

builtin_macros_coerce_pointee_requires_one_generic = `CoercePointee` can only be derived on `struct`s that are generic over at least one type

builtin_macros_coerce_pointee_requires_one_pointee = exactly one generic type parameter must be marked as `#[pointee]` to derive `CoercePointee` traits

builtin_macros_coerce_pointee_requires_transparent = `CoercePointee` can only be derived on `struct`s with `#[repr(transparent)]`

builtin_macros_coerce_pointee_too_many_pointees = only one type parameter can be marked as `#[pointee]` when deriving `CoercePointee` traits
    .label = here another type parameter is marked as `#[pointee]`


builtin_macros_concat_bytes_array = cannot concatenate doubly nested array
    .note = byte strings are treated as arrays of bytes
    .help = try flattening the array

builtin_macros_concat_bytes_bad_repeat = repeat count is not a positive number

builtin_macros_concat_bytes_invalid = cannot concatenate {$lit_kind} literals
    .byte_char = try using a byte character
    .byte_str = try using a byte string
    .c_str = try using a null-terminated byte string
    .c_str_note = concatenating C strings is ambiguous about including the '\0'
    .number_array = try wrapping the number in an array

builtin_macros_concat_bytes_missing_literal = expected a byte literal
    .note = only byte literals (like `b"foo"`, `b's'` and `[3, 4, 5]`) can be passed to `concat_bytes!()`

builtin_macros_concat_bytes_non_u8 = numeric literal is not a `u8`

builtin_macros_concat_bytes_oob = numeric literal is out of bounds

builtin_macros_concat_bytestr = cannot concatenate a byte string literal
builtin_macros_concat_c_str_lit = cannot concatenate a C string literal

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
    .cargo = Cargo sets build script variables at run time. Use `std::env::var({$var_expr})` instead
    .custom = use `std::env::var({$var_expr})` to read the variable at run time

builtin_macros_env_not_unicode = environment variable `{$var}` is not a valid Unicode string

builtin_macros_env_takes_args = `env!()` takes 1 or 2 arguments

builtin_macros_expected_comma_in_list = expected token: `,`

builtin_macros_expected_one_cfg_pattern = expected 1 cfg-pattern

builtin_macros_expected_other = expected operand, {$is_inline_asm ->
    [false] options
    *[true] clobber_abi, options
    }, or additional template string

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

builtin_macros_format_redundant_args = redundant {$n ->
    [one] argument
    *[more] arguments
    }
    .help = {$n ->
        [one] the formatting string already captures the binding directly, it doesn't need to be included in the argument list
        *[more] the formatting strings already captures the bindings directly, they don't need to be included in the argument list
    }
    .note = {$n ->
        [one] the formatting specifier is referencing the binding already
        *[more] the formatting specifiers are referencing the bindings already
    }
    .suggestion = this can be removed

builtin_macros_format_remove_raw_ident = remove the `r#`

builtin_macros_format_reorder_format_parameter = did you mean `{$replacement}`?

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

builtin_macros_format_use_positional = consider using a positional formatting argument instead

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

builtin_macros_naked_functions_testing_attribute =
    cannot use `#[unsafe(naked)]` with testing attributes
    .label = function marked with testing attribute here
    .naked_attribute = `#[unsafe(naked)]` is incompatible with testing attributes

builtin_macros_no_default_variant = `#[derive(Default)]` on enum with no `#[default]`
    .label = this enum needs a unit variant marked with `#[default]`
    .suggestion = make this unit variant default by placing `#[default]` on it

builtin_macros_non_exhaustive_default = default variant must be exhaustive
    .label = declared `#[non_exhaustive]` here
    .help = consider a manual implementation of `Default`

builtin_macros_non_generic_pointee = the `#[pointee]` attribute may only be used on generic parameters

builtin_macros_non_unit_default = the `#[default]` attribute may only be used on unit enum variants{$post}
    .help = consider a manual implementation of `Default`

builtin_macros_only_one_argument = {$name} takes 1 argument

builtin_macros_proc_macro = `proc-macro` crate types currently cannot export any items other than functions tagged with `#[proc_macro]`, `#[proc_macro_derive]`, or `#[proc_macro_attribute]`

builtin_macros_proc_macro_attribute_only_be_used_on_bare_functions = the `#[{$path}]` attribute may only be used on bare functions

builtin_macros_proc_macro_attribute_only_usable_with_crate_type = the `#[{$path}]` attribute is only usable with crates of the `proc-macro` crate type

builtin_macros_requires_cfg_pattern =
    macro requires a cfg-pattern as an argument
    .label = cfg-pattern required

builtin_macros_source_uitls_expected_item = expected item, found `{$token}`

builtin_macros_takes_no_arguments = {$name} takes no arguments

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
