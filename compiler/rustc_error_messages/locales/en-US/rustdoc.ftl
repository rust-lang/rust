rustdoc_compilation_failed =
    Compilation failed, aborting rustdoc

rustdoc_couldnt_generate_documentation =
    couldn't generate documentation: {$error}
    .note = failed to create or modify "{$file}"

rustdoc_missing_crate_level_docs =
    no documentation found for this crate's top-level module
    .help = The following guide may be of use:
            {$doc_rust_lang_org_channel}/rustdoc/how-to-write-documentation.html

rustdoc_deprecated_attr =
    the `#![doc({$attr_name})]` attribute is deprecated
    .note = see issue #44136 <https://github.com/rust-lang/rust/issues/44136> for more information

rustdoc_deprecated_attr_no_default_passes =
    `#![doc(no_default_passes)]` no longer functions; you may want to use `#![doc(document_private_items)]`

rustdoc_deprecated_attr_passes =
    `#![doc(passes = "...")]` no longer functions; you may want to use `#![doc(document_private_items)]`

rustdoc_deprecated_attr_plugins =
    `#![doc(plugins = "...")]` no longer functions; see CVE-2018-1000622 <https://nvd.nist.gov/vuln/detail/CVE-2018-1000622>

rustdoc_could_not_resolve_path =
    failed to resolve: could not resolve path `{$path}`
    .label = could not resolve path `{$path}`
    .note = this error was originally ignored because you are running `rustdoc`
    .help = try running again with `rustc` or `cargo check` and you may get a more detailed error

rustdoc_unrecognized_emission_type =
    unrecognized emission type: {$kind}

rustdoc_invalid_extern_html_root_url =
    --extern-html-root-url must be of the form name=url

rustdoc_missing_file_operand =
    missing file operand

rustdoc_too_many_file_operands =
    too many file operands

rustdoc_no_run_flag_without_test_flag =
    the `--test` flag must be passed to enable `--no-run`

rustdoc_cannot_use_out_dir_and_output_flags =
    cannot use both 'out-dir' and 'output' at once

rustdoc_option_extend_css_arg_not_file =
    option --extend-css argument must be a file

rustdoc_theme_arg_not_file =
    invalid argument: "{$theme_arg}"
    .help = arguments to --theme must be files

rustdoc_theme_arg_not_css_file =
    invalid argument: "{$theme_arg}"
    .help = arguments to --theme must have a .css extension

rustdoc_error_loading_theme_file =
    error loading theme file: "{$theme_arg}"

rustdoc_theme_file_missing_default_theme_css_rules =
    theme file "{$theme_arg}" is missing CSS rules from the default theme
    .warn = the theme may appear incorrect when loaded
    .help = "to see what rules are missing, call `rustdoc --check-theme "{$theme_arg}"`

rustdoc_unknown_input_format =
    unkown input format: {$theme_arg}

rustdoc_index_page_arg_not_file =
    option `--index-page` argument must be a file

rustdoc_unknown_crate_type =
    unknown crate type: {$error}

rustdoc_html_output_not_supported_with_show_coverage_flag =
    html output format isn't supported for the --show-coverage option

rustdoc_generate_link_to_definition_flag_not_with_html_output_format =
    --generate-link-to-definition option can only be used with HTML output format

rustdoc_scrape_examples_output_path_and_target_crate_not_used_together =
    must use --scrape-examples-output-path and --scrape-examples-target-crate together

rustdoc_scrape_tests_not_with_scrape_examples_output_path_and_target_crate =
    must use --scrape-examples-output-path and --scrape-examples-target-crate with --scrape-tests

rustdoc_flag_deprecated =
    the `{$flag}` flag is deprecated
    .note = see issue #44136 <https://github.com/rust-lang/rust/issues/44136> for more information

rustdoc_flag_removed =
    the `{$flag}` flag no longer functions
    .note = see issue #44136 <https://github.com/rust-lang/rust/issues/44136> for more information

rustdoc_use_document_private_items_flag =
    you may want to use --document-private-items

rustdoc_see_rustdoc_plugins_cve =
    see CVE-2018-1000622

rustdoc_error_reading_file =
    error reading `{$file_path}`: {$error}

rustdoc_error_reading_file_not_utf8 =
    error reading `{$file_path}`: not UTF-8

rustdoc_error_loading_examples =
    failed to load examples: {$error} (for path {$path})

rustdoc_anonymous_imports_cannot_be_inlined =
    anonymous imports cannot be inlined
    .import_span = anonymous import

rustdoc_invalid_codeblock_attribute =
    unknown attribute `{$attr_name}`. Did you mean `{$suggested_attr_name}`?
    .compile_fail = the code block will either not be tested if not marked as a rust one or won't fail if it compiles successfully
    .should_panic = the code block will either not be tested if not marked as a rust one or won't fail if it doesn't panic when running
    .no_run = the code block will either not be tested if not marked as a rust one or will be run (which you might not want)
    .test_harness = the code block will either not be tested if not marked as a rust one or the code will be wrapped inside a main function

rustdoc_failed_to_read_file =
    failed to read file {$path}: {$error}

rustdoc_bare_url_not_hyperlink =
    this URL is not a hyperlink
    .note = bare URLs are not automatically turned into clickable links
    .suggestion = use an automatic link instead

rustdoc_missing_doc_code_examples =
    missing code example in this documentation

rustdoc_private_doc_tests =
    documentation test in private item

rustdoc_cfg_unexpected_literal =
    unexpected literal

rustdoc_cfg_expected_single_identifier =
    expected a single identifier

rustdoc_cfg_option_value_not_string_literal =
    value of cfg option should be a string literal

rustdoc_cfg_expected_one_cfg_pattern =
    expected 1 cfg-pattern

rustdoc_cfg_invalid_predicate =
    invalid predicate

rustdoc_unclosed_html_tag =
    unclosed HTML tag `{$tag}`

rustdoc_unclosed_html_comment =
    Unclosed HTML comment

rustdoc_mark_source_code =
    try marking as source code

rustdoc_unopened_html_tag =
    unopened HTML tag `{$tag_name}`

rustdoc_unclosed_quoted_html_attribute =
    unclosed quoted HTML attribute on tag `{$tag_name}`

rustdoc_invalid_self_closing_html_tag =
    invalid self-closing HTML tag `{$tag_name}`
