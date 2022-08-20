interface_ferris_identifier =
    Ferris cannot be used as an identifier
    .suggestion = try using their name instead

interface_emoji_identifier =
    identifiers cannot contain emoji: `{$ident}`

mixed_bin_crate =
    cannot mix `bin` crate type with others

mixed_proc_macro_crate =
    cannot mix `proc-macro` crate type with others

proc_macro_doc_without_arg =
    Trying to document proc macro crate without passing '--crate-type proc-macro to rustdoc
    .warn = The generated documentation may be incorrect

error_writing_dependencies =
    error writing dependencies to `{$path}`: {$error}

input_file_would_be_overwritten =
    the input file "{$path}" would be overwritten by the generated executable

generated_file_conflicts_with_directory =
    the generated executable for the input file "{$input_path}" conflicts with the existing directory "{$dir_path}"

temps_dir_error =
    failed to find or create the directory specified by `--temps-dir`

out_dir_error =
    failed to find or create the directory specified by `--out-dir`

cant_emit_mir =
    could not emit MIR: {$error}

rustc_error_fatal =
    fatal error triggered by #[rustc_error]

rustc_error_unexpected_annotation =
    unexpected annotation used with `#[rustc_error(...)]!

failed_writing_file =
    failed to write file {$path}: {$error}"
