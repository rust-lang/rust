interface_abi_required_feature =
    target feature `{$feature}` must be {$enabled} to ensure that the ABI of the current target can be implemented correctly
    .note = this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
interface_abi_required_feature_issue = for more information, see issue #116344 <https://github.com/rust-lang/rust/issues/116344>

interface_crate_name_does_not_match = `--crate-name` and `#[crate_name]` are required to match, but `{$crate_name}` != `{$attr_crate_name}`

interface_crate_name_invalid = crate names cannot start with a `-`, but `{$crate_name}` has a leading hyphen

interface_emoji_identifier =
    identifiers cannot contain emoji: `{$ident}`

interface_error_writing_dependencies =
    error writing dependencies to `{$path}`: {$error}

interface_failed_writing_file =
    failed to write file {$path}: {$error}"

interface_ferris_identifier =
    Ferris cannot be used as an identifier
    .suggestion = try using their name instead

interface_generated_file_conflicts_with_directory =
    the generated executable for the input file "{$input_path}" conflicts with the existing directory "{$dir_path}"

interface_ignoring_extra_filename = ignoring -C extra-filename flag due to -o flag

interface_ignoring_out_dir = ignoring --out-dir flag due to -o flag

interface_input_file_would_be_overwritten =
    the input file "{$path}" would be overwritten by the generated executable

interface_invalid_crate_type_value = invalid `crate_type` value
    .suggestion = did you mean

interface_mixed_bin_crate =
    cannot mix `bin` crate type with others

interface_mixed_proc_macro_crate =
    cannot mix `proc-macro` crate type with others

interface_multiple_output_types_adaption =
    due to multiple output types requested, the explicitly specified output file name will be adapted for each output type

interface_multiple_output_types_to_stdout = can't use option `-o` or `--emit` to write multiple output types to stdout
interface_out_dir_error =
    failed to find or create the directory specified by `--out-dir`

interface_proc_macro_crate_panic_abort =
    building proc macro crate with `panic=abort` or `panic=immediate-abort` may crash the compiler should the proc-macro panic

interface_temps_dir_error =
    failed to find or create the directory specified by `--temps-dir`
