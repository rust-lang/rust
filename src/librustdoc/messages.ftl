rustdoc_couldnt_generate_documentation =
    couldn't generate documentation: {$error}

rustdoc_failed_to_create_or_modify_file = failed to create or modify "{$file}"

rustdoc_compilation_failed = compilation failed, aborting rustdoc

rustdoc_load_string_error_read_fail = error reading `{$file_path}`: {$err}

rustdoc_load_string_error_bad_utf8 = error reading `{$file_path}`: not UTF-8

rustdoc_arguments_to_theme_must_be_files = invalid argument: "{$theme}"
    .help = arguments to --theme must be files

rustdoc_arguments_to_theme_must_have_a_css_extension = invalid argument: "{$theme}"
    .help = arguments to --theme must have a .css extension

rustdoc_error_loading_theme_file = error loading theme file: "{$theme}"

rustdoc_theme_file_missing_css_rules_from_default_theme = theme file "{$theme}" is missing CSS rules from the default theme
    .warn = the theme may appear incorrect when loaded
    .help = to see what rules are missing, call `rustdoc --check-theme "{$theme}"`

rustdoc_scrape_examples_must_use_output_path_and_target_crate_together = must use --scrape-examples-output-path and --scrape-examples-target-crate together

rustdoc_scrape_examples_must_use_output_path_and_target_crate_with_scrape_tests = must use --scrape-examples-output-path and --scrape-examples-target-crate with --scrape-tests

rustdoc_load_examples_failed = failed to load examples: {$err}

rustdoc_unrecognized_emission_type = unrecognized emission type: {$kind}

rustdoc_missing_file_operand = missing file operand

rustdoc_too_many_file_operands = too many file operands

rustdoc_test_flag_must_be_passed_to_enable_no_run = the `--test` flag must be passed to enable `--no-run`

rustdoc_cannot_use_both_out_dir_and_output_at_once = cannot use both 'out-dir' and 'output' at once

rustdoc_extend_css_arg_must_be_a_file = option --extend-css argument must be a file

rustdoc_unknown_input_format = unknown input format: {$format}

rustdoc_index_page_arg_must_be_a_file = option `--index-page` argument must be a file

rustdoc_unknown_crate_type = unknown crate type: {$err}

rustdoc_html_output_format_unsupported_for_show_coverage_option = html output format isn't supported for the --show-coverage option

rustdoc_generate_link_to_definition_option_can_only_be_used_with_html_output_format = --generate-link-to-definition option can only be used with HTML output format

rustdoc_flag_is_deprecated = the `{$flag}` flag is deprecated
    .note =
        see issue #44136 <https://github.com/rust-lang/rust/issues/44136>
        for more information,

rustdoc_flag_no_longer_functions = the `{$flag}` flag no longer functions
    .note =
        see issue #44136 <https://github.com/rust-lang/rust/issues/44136>
        for more information,

rustdoc_may_want_to_use_document_private_items = you may want to use --document-private-items

rustdoc_see_cve_2018_1000622 = see CVE-2018-1000622
