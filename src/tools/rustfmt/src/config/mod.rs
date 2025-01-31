use std::cell::Cell;
use std::fs::File;
use std::io::{Error, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::{env, fs};

use thiserror::Error;

use crate::config::config_type::ConfigType;
#[allow(unreachable_pub)]
pub use crate::config::file_lines::{FileLines, FileName, Range};
#[allow(unreachable_pub)]
pub use crate::config::macro_names::MacroSelector;
#[allow(unreachable_pub)]
pub use crate::config::options::*;

#[macro_use]
pub(crate) mod config_type;
#[macro_use]
#[allow(unreachable_pub)]
pub(crate) mod options;

pub(crate) mod file_lines;
#[allow(unreachable_pub)]
pub(crate) mod lists;
pub(crate) mod macro_names;
pub(crate) mod style_edition;

// This macro defines configuration options used in rustfmt. Each option
// is defined as follows:
//
// `name: value type, is stable, description;`
create_config! {
    // Fundamental stuff
    max_width: MaxWidth, true, "Maximum width of each line";
    hard_tabs: HardTabs, true, "Use tab characters for indentation, spaces for alignment";
    tab_spaces: TabSpaces, true, "Number of spaces per tab";
    newline_style: NewlineStyleConfig, true, "Unix or Windows line endings";
    indent_style: IndentStyleConfig, false, "How do we indent expressions or items";

    // Width Heuristics
    use_small_heuristics: UseSmallHeuristics, true, "Whether to use different \
        formatting for items and expressions if they satisfy a heuristic notion of 'small'";
    width_heuristics: WidthHeuristicsConfig, false, "'small' heuristic values";
    fn_call_width: FnCallWidth, true, "Maximum width of the args of a function call before \
        falling back to vertical formatting.";
    attr_fn_like_width: AttrFnLikeWidth, true, "Maximum width of the args of a function-like \
        attributes before falling back to vertical formatting.";
    struct_lit_width: StructLitWidth, true, "Maximum width in the body of a struct lit before \
        falling back to vertical formatting.";
    struct_variant_width: StructVariantWidth, true, "Maximum width in the body of a struct variant \
        before falling back to vertical formatting.";
    array_width: ArrayWidth, true,  "Maximum width of an array literal before falling \
        back to vertical formatting.";
    chain_width: ChainWidth, true, "Maximum length of a chain to fit on a single line.";
    single_line_if_else_max_width: SingleLineIfElseMaxWidth, true, "Maximum line length for single \
        line if-else expressions. A value of zero means always break if-else expressions.";
    single_line_let_else_max_width: SingleLineLetElseMaxWidth, true, "Maximum line length for \
        single line let-else statements. A value of zero means always format the divergent `else` \
        block over multiple lines.";

    // Comments. macros, and strings
    wrap_comments: WrapComments, false, "Break comments to fit on the line";
    format_code_in_doc_comments: FormatCodeInDocComments, false, "Format the code snippet in \
        doc comments.";
    doc_comment_code_block_width: DocCommentCodeBlockWidth, false, "Maximum width for code \
        snippets in doc comments. No effect unless format_code_in_doc_comments = true";
    comment_width: CommentWidth, false,
        "Maximum length of comments. No effect unless wrap_comments = true";
    normalize_comments: NormalizeComments, false, "Convert /* */ comments to // comments where \
        possible";
    normalize_doc_attributes: NormalizeDocAttributes, false, "Normalize doc attributes as doc \
        comments";
    format_strings: FormatStrings, false, "Format string literals where necessary";
    format_macro_matchers: FormatMacroMatchers, false,
        "Format the metavariable matching patterns in macros";
    format_macro_bodies: FormatMacroBodies, false,
        "Format the bodies of declarative macro definitions";
    skip_macro_invocations: SkipMacroInvocations, false,
        "Skip formatting the bodies of macros invoked with the following names.";
    hex_literal_case: HexLiteralCaseConfig, false, "Format hexadecimal integer literals";

    // Single line expressions and items
    empty_item_single_line: EmptyItemSingleLine, false,
        "Put empty-body functions and impls on a single line";
    struct_lit_single_line: StructLitSingleLine, false,
        "Put small struct literals on a single line";
    fn_single_line: FnSingleLine, false, "Put single-expression functions on a single line";
    where_single_line: WhereSingleLine, false, "Force where-clauses to be on a single line";

    // Imports
    imports_indent: ImportsIndent, false, "Indent of imports";
    imports_layout: ImportsLayout, false, "Item layout inside a import block";
    imports_granularity: ImportsGranularityConfig, false,
        "Merge or split imports to the provided granularity";
    group_imports: GroupImportsTacticConfig, false,
        "Controls the strategy for how imports are grouped together";
    merge_imports: MergeImports, false, "(deprecated: use imports_granularity instead)";

    // Ordering
    reorder_imports: ReorderImports, true, "Reorder import and extern crate statements \
        alphabetically";
    reorder_modules: ReorderModules, true, "Reorder module statements alphabetically in group";
    reorder_impl_items: ReorderImplItems, false, "Reorder impl items";

    // Spaces around punctuation
    type_punctuation_density: TypePunctuationDensity, false,
        "Determines if '+' or '=' are wrapped in spaces in the punctuation of types";
    space_before_colon: SpaceBeforeColon, false, "Leave a space before the colon";
    space_after_colon: SpaceAfterColon, false, "Leave a space after the colon";
    spaces_around_ranges: SpacesAroundRanges, false, "Put spaces around the  .. and ..= range \
        operators";
    binop_separator: BinopSeparator, false,
        "Where to put a binary operator when a binary expression goes multiline";

    // Misc.
    remove_nested_parens: RemoveNestedParens, true, "Remove nested parens";
    combine_control_expr: CombineControlExpr, false, "Combine control expressions with function \
        calls";
    short_array_element_width_threshold: ShortArrayElementWidthThreshold, true,
        "Width threshold for an array element to be considered short";
    overflow_delimited_expr: OverflowDelimitedExpr, false,
        "Allow trailing bracket/brace delimited expressions to overflow";
    struct_field_align_threshold: StructFieldAlignThreshold, false,
        "Align struct fields if their diffs fits within threshold";
    enum_discrim_align_threshold: EnumDiscrimAlignThreshold, false,
        "Align enum variants discrims, if their diffs fit within threshold";
    match_arm_blocks: MatchArmBlocks, false, "Wrap the body of arms in blocks when it does not fit \
        on the same line with the pattern of arms";
    match_arm_leading_pipes: MatchArmLeadingPipeConfig, true,
        "Determines whether leading pipes are emitted on match arms";
    force_multiline_blocks: ForceMultilineBlocks, false,
        "Force multiline closure bodies and match arms to be wrapped in a block";
    fn_args_layout: FnArgsLayout, true,
        "(deprecated: use fn_params_layout instead)";
    fn_params_layout: FnParamsLayout, true,
        "Control the layout of parameters in function signatures.";
    brace_style: BraceStyleConfig, false, "Brace style for items";
    control_brace_style: ControlBraceStyleConfig, false,
        "Brace style for control flow constructs";
    trailing_semicolon: TrailingSemicolon, false,
        "Add trailing semicolon after break, continue and return";
    trailing_comma: TrailingComma, false,
        "How to handle trailing commas for lists";
    match_block_trailing_comma: MatchBlockTrailingComma, true,
        "Put a trailing comma after a block based match arm (non-block arms are not affected)";
    blank_lines_upper_bound: BlankLinesUpperBound, false,
        "Maximum number of blank lines which can be put between items";
    blank_lines_lower_bound: BlankLinesLowerBound, false,
        "Minimum number of blank lines which must be put between items";
    edition: EditionConfig, true, "The edition of the parser (RFC 2052)";
    style_edition: StyleEditionConfig, true, "The edition of the Style Guide (RFC 3338)";
    version: VersionConfig, false, "Version of formatting rules";
    inline_attribute_width: InlineAttributeWidth, false,
        "Write an item and its attribute on the same line \
        if their combined width is below a threshold";
    format_generated_files: FormatGeneratedFiles, false, "Format generated files";
    generated_marker_line_search_limit: GeneratedMarkerLineSearchLimit, false, "Number of lines to \
        check for a `@generated` marker when `format_generated_files` is enabled";

    // Options that can change the source code beyond whitespace/blocks (somewhat linty things)
    merge_derives: MergeDerives, true, "Merge multiple `#[derive(...)]` into a single one";
    use_try_shorthand: UseTryShorthand, true, "Replace uses of the try! macro by the ? shorthand";
    use_field_init_shorthand: UseFieldInitShorthand, true, "Use field initialization shorthand if \
        possible";
    force_explicit_abi: ForceExplicitAbi, true, "Always print the abi for extern items";
    condense_wildcard_suffixes: CondenseWildcardSuffixes, false, "Replace strings of _ wildcards \
        by a single .. in tuple patterns";

    // Control options (changes the operation of rustfmt, rather than the formatting)
    color: ColorConfig, false,
        "What Color option to use when none is supplied: Always, Never, Auto";
    required_version: RequiredVersion, false,
        "Require a specific version of rustfmt";
    unstable_features: UnstableFeatures, false,
            "Enables unstable features. Only available on nightly channel";
    disable_all_formatting: DisableAllFormatting, true, "Don't reformat anything";
    skip_children: SkipChildren, false, "Don't reformat out of line modules";
    hide_parse_errors: HideParseErrors, false, "Hide errors from the parser";
    show_parse_errors: ShowParseErrors, false, "Show errors from the parser (unstable)";
    error_on_line_overflow: ErrorOnLineOverflow, false, "Error if unable to get all lines within \
        max_width";
    error_on_unformatted: ErrorOnUnformatted, false,
        "Error if unable to get comments or string literals within max_width, \
         or they are left with trailing whitespaces";
    ignore: Ignore, false,
        "Skip formatting the specified files and directories";

    // Not user-facing
    verbose: Verbose, false, "How much to information to emit to the user";
    file_lines: FileLinesConfig, false,
        "Lines to format; this is not supported in rustfmt.toml, and can only be specified \
         via the --file-lines option";
    emit_mode: EmitModeConfig, false,
        "What emit Mode to use when none is supplied";
    make_backup: MakeBackup, false, "Backup changed files";
    print_misformatted_file_names: PrintMisformattedFileNames, true,
        "Prints the names of mismatched files that were formatted. Prints the names of \
         files that would be formatted when used with `--check` mode. ";
}

#[derive(Error, Debug)]
#[error("Could not output config: {0}")]
pub struct ToTomlError(toml::ser::Error);

impl PartialConfig {
    pub fn to_toml(&self) -> Result<String, ToTomlError> {
        // Non-user-facing options can't be specified in TOML
        let mut cloned = self.clone();
        cloned.file_lines = None;
        cloned.verbose = None;
        cloned.width_heuristics = None;
        cloned.print_misformatted_file_names = None;
        cloned.merge_imports = None;
        cloned.fn_args_layout = None;
        cloned.hide_parse_errors = None;

        ::toml::to_string(&cloned).map_err(ToTomlError)
    }

    pub(super) fn to_parsed_config(
        self,
        style_edition_override: Option<StyleEdition>,
        edition_override: Option<Edition>,
        version_override: Option<Version>,
        dir: &Path,
    ) -> Config {
        Config::default_for_possible_style_edition(
            style_edition_override.or(self.style_edition),
            edition_override.or(self.edition),
            version_override.or(self.version),
        )
        .fill_from_parsed_config(self, dir)
    }
}

impl Config {
    pub fn default_for_possible_style_edition(
        style_edition: Option<StyleEdition>,
        edition: Option<Edition>,
        version: Option<Version>,
    ) -> Config {
        // Ensures the configuration defaults associated with Style Editions
        // follow the precedence set in
        // https://rust-lang.github.io/rfcs/3338-style-evolution.html
        // 'version' is a legacy alias for 'style_edition' that we'll support
        // for some period of time
        // FIXME(calebcartwright) - remove 'version' at some point
        match (style_edition, version, edition) {
            (Some(se), _, _) => Self::default_with_style_edition(se),
            (None, Some(Version::Two), _) => {
                Self::default_with_style_edition(StyleEdition::Edition2024)
            }
            (None, Some(Version::One), _) => {
                Self::default_with_style_edition(StyleEdition::Edition2015)
            }
            (None, None, Some(e)) => Self::default_with_style_edition(e.into()),
            (None, None, None) => Config::default(),
        }
    }

    pub(crate) fn version_meets_requirement(&self) -> bool {
        if self.was_set().required_version() {
            let version = env!("CARGO_PKG_VERSION");
            let required_version = self.required_version();
            if version != required_version {
                println!(
                    "Error: rustfmt version ({version}) doesn't match the required version \
({required_version})"
                );
                return false;
            }
        }

        true
    }

    /// Constructs a `Config` from the toml file specified at `file_path`.
    ///
    /// This method only looks at the provided path, for a method that
    /// searches parents for a `rustfmt.toml` see `from_resolved_toml_path`.
    ///
    /// Returns a `Config` if the config could be read and parsed from
    /// the file, otherwise errors.
    pub(super) fn from_toml_path(
        file_path: &Path,
        edition: Option<Edition>,
        style_edition: Option<StyleEdition>,
        version: Option<Version>,
    ) -> Result<Config, Error> {
        let mut file = File::open(&file_path)?;
        let mut toml = String::new();
        file.read_to_string(&mut toml)?;
        Config::from_toml_for_style_edition(&toml, file_path, edition, style_edition, version)
            .map_err(|err| Error::new(ErrorKind::InvalidData, err))
    }

    /// Resolves the config for input in `dir`.
    ///
    /// Searches for `rustfmt.toml` beginning with `dir`, and
    /// recursively checking parents of `dir` if no config file is found.
    /// If no config file exists in `dir` or in any parent, a
    /// default `Config` will be returned (and the returned path will be empty).
    ///
    /// Returns the `Config` to use, and the path of the project file if there was
    /// one.
    pub(super) fn from_resolved_toml_path(
        dir: &Path,
        edition: Option<Edition>,
        style_edition: Option<StyleEdition>,
        version: Option<Version>,
    ) -> Result<(Config, Option<PathBuf>), Error> {
        /// Try to find a project file in the given directory and its parents.
        /// Returns the path of the nearest project file if one exists,
        /// or `None` if no project file was found.
        fn resolve_project_file(dir: &Path) -> Result<Option<PathBuf>, Error> {
            let mut current = if dir.is_relative() {
                env::current_dir()?.join(dir)
            } else {
                dir.to_path_buf()
            };

            current = fs::canonicalize(current)?;

            loop {
                match get_toml_path(&current) {
                    Ok(Some(path)) => return Ok(Some(path)),
                    Err(e) => return Err(e),
                    _ => (),
                }

                // If the current directory has no parent, we're done searching.
                if !current.pop() {
                    break;
                }
            }

            // If nothing was found, check in the home directory.
            if let Some(home_dir) = dirs::home_dir() {
                if let Some(path) = get_toml_path(&home_dir)? {
                    return Ok(Some(path));
                }
            }

            // If none was found there either, check in the user's configuration directory.
            if let Some(mut config_dir) = dirs::config_dir() {
                config_dir.push("rustfmt");
                if let Some(path) = get_toml_path(&config_dir)? {
                    return Ok(Some(path));
                }
            }

            Ok(None)
        }

        match resolve_project_file(dir)? {
            None => Ok((
                Config::default_for_possible_style_edition(style_edition, edition, version),
                None,
            )),
            Some(path) => Config::from_toml_path(&path, edition, style_edition, version)
                .map(|config| (config, Some(path))),
        }
    }

    #[allow(dead_code)]
    pub(super) fn from_toml(toml: &str, file_path: &Path) -> Result<Config, String> {
        Self::from_toml_for_style_edition(toml, file_path, None, None, None)
    }

    pub(crate) fn from_toml_for_style_edition(
        toml: &str,
        file_path: &Path,
        edition: Option<Edition>,
        style_edition: Option<StyleEdition>,
        version: Option<Version>,
    ) -> Result<Config, String> {
        let parsed: ::toml::Value = toml
            .parse()
            .map_err(|e| format!("Could not parse TOML: {}", e))?;
        let mut err = String::new();
        let table = parsed
            .as_table()
            .ok_or_else(|| String::from("Parsed config was not table"))?;
        for key in table.keys() {
            if !Config::is_valid_name(key) {
                let msg = &format!("Warning: Unknown configuration option `{key}`\n");
                err.push_str(msg)
            }
        }

        match parsed.try_into::<PartialConfig>() {
            Ok(parsed_config) => {
                if !err.is_empty() {
                    eprint!("{err}");
                }
                let dir = file_path.parent().ok_or_else(|| {
                    format!("failed to get parent directory for {}", file_path.display())
                })?;

                Ok(parsed_config.to_parsed_config(style_edition, edition, version, dir))
            }
            Err(e) => {
                let err_msg = format!(
                    "The file `{}` failed to parse.\nError details: {e}",
                    file_path.display()
                );
                err.push_str(&err_msg);
                Err(err_msg)
            }
        }
    }
}

/// Loads a config by checking the client-supplied options and if appropriate, the
/// file system (including searching the file system for overrides).
pub fn load_config<O: CliOptions>(
    file_path: Option<&Path>,
    options: Option<O>,
) -> Result<(Config, Option<PathBuf>), Error> {
    let (over_ride, edition, style_edition, version) = match options {
        Some(ref opts) => (
            config_path(opts)?,
            opts.edition(),
            opts.style_edition(),
            opts.version(),
        ),
        None => (None, None, None, None),
    };

    let result = if let Some(over_ride) = over_ride {
        Config::from_toml_path(over_ride.as_ref(), edition, style_edition, version)
            .map(|p| (p, Some(over_ride.to_owned())))
    } else if let Some(file_path) = file_path {
        Config::from_resolved_toml_path(file_path, edition, style_edition, version)
    } else {
        Ok((
            Config::default_for_possible_style_edition(style_edition, edition, version),
            None,
        ))
    };

    result.map(|(mut c, p)| {
        if let Some(options) = options {
            options.apply_to(&mut c);
        }
        (c, p)
    })
}

// Check for the presence of known config file names (`rustfmt.toml`, `.rustfmt.toml`) in `dir`
//
// Return the path if a config file exists, empty if no file exists, and Error for IO errors
fn get_toml_path(dir: &Path) -> Result<Option<PathBuf>, Error> {
    const CONFIG_FILE_NAMES: [&str; 2] = [".rustfmt.toml", "rustfmt.toml"];
    for config_file_name in &CONFIG_FILE_NAMES {
        let config_file = dir.join(config_file_name);
        match fs::metadata(&config_file) {
            // Only return if it's a file to handle the unlikely situation of a directory named
            // `rustfmt.toml`.
            Ok(ref md) if md.is_file() => return Ok(Some(config_file.canonicalize()?)),
            // Return the error if it's something other than `NotFound`; otherwise we didn't
            // find the project file yet, and continue searching.
            Err(e) => {
                if e.kind() != ErrorKind::NotFound {
                    let ctx = format!("Failed to get metadata for config file {:?}", &config_file);
                    let err = anyhow::Error::new(e).context(ctx);
                    return Err(Error::new(ErrorKind::Other, err));
                }
            }
            _ => {}
        }
    }
    Ok(None)
}

fn config_path(options: &dyn CliOptions) -> Result<Option<PathBuf>, Error> {
    let config_path_not_found = |path: &str| -> Result<Option<PathBuf>, Error> {
        Err(Error::new(
            ErrorKind::NotFound,
            format!(
                "Error: unable to find a config file for the given path: `{}`",
                path
            ),
        ))
    };

    // Read the config_path and convert to parent dir if a file is provided.
    // If a config file cannot be found from the given path, return error.
    match options.config_path() {
        Some(path) if !path.exists() => config_path_not_found(path.to_str().unwrap()),
        Some(path) if path.is_dir() => {
            let config_file_path = get_toml_path(path)?;
            if config_file_path.is_some() {
                Ok(config_file_path)
            } else {
                config_path_not_found(path.to_str().unwrap())
            }
        }
        Some(path) => Ok(Some(
            // Canonicalize only after checking above that the `path.exists()`.
            path.canonicalize()?,
        )),
        None => Ok(None),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::str;

    use crate::config::macro_names::{MacroName, MacroSelectors};
    use rustfmt_config_proc_macro::{nightly_only_test, stable_only_test};

    #[allow(dead_code)]
    mod mock {
        use super::super::*;
        use crate::config_option_with_style_edition_default;
        use rustfmt_config_proc_macro::config_type;

        #[config_type]
        pub(crate) enum PartiallyUnstableOption {
            V1,
            V2,
            #[unstable_variant]
            V3,
        }

        config_option_with_style_edition_default!(
            StableOption, bool, _ => false;
            UnstableOption, bool, _ => false;
            PartiallyUnstable, PartiallyUnstableOption, _ => PartiallyUnstableOption::V1;
        );

        create_config! {
            // Options that are used by the generated functions
            max_width: MaxWidth, true, "Maximum width of each line";
            required_version: RequiredVersion, false, "Require a specific version of rustfmt.";
            ignore: Ignore, false, "Skip formatting the specified files and directories.";
            verbose: Verbose, false, "How much to information to emit to the user";
            file_lines: FileLinesConfig, false,
                "Lines to format; this is not supported in rustfmt.toml, and can only be specified \
                    via the --file-lines option";

            // merge_imports deprecation
            imports_granularity: ImportsGranularityConfig, false, "Merge imports";
            merge_imports: MergeImports, false, "(deprecated: use imports_granularity instead)";

            // fn_args_layout renamed to fn_params_layout
            fn_args_layout: FnArgsLayout, true, "(deprecated: use fn_params_layout instead)";
            fn_params_layout: FnParamsLayout, true,
                "Control the layout of parameters in a function signatures.";

            // hide_parse_errors renamed to show_parse_errors
            hide_parse_errors: HideParseErrors, false,
                "(deprecated: use show_parse_errors instead)";
            show_parse_errors: ShowParseErrors, false,
                "Show errors from the parser (unstable)";


            // Width Heuristics
            use_small_heuristics: UseSmallHeuristics, true,
                "Whether to use different formatting for items and \
                 expressions if they satisfy a heuristic notion of 'small'.";
            width_heuristics: WidthHeuristicsConfig, false, "'small' heuristic values";

            fn_call_width: FnCallWidth, true, "Maximum width of the args of a function call before \
                falling back to vertical formatting.";
            attr_fn_like_width: AttrFnLikeWidth, true, "Maximum width of the args of a \
                function-like attributes before falling back to vertical formatting.";
            struct_lit_width: StructLitWidth, true, "Maximum width in the body of a struct lit \
                before falling back to vertical formatting.";
            struct_variant_width: StructVariantWidth, true, "Maximum width in the body of a struct \
                variant before falling back to vertical formatting.";
            array_width: ArrayWidth, true,  "Maximum width of an array literal before falling \
                back to vertical formatting.";
            chain_width: ChainWidth, true, "Maximum length of a chain to fit on a single line.";
            single_line_if_else_max_width: SingleLineIfElseMaxWidth, true, "Maximum line length \
                for single line if-else expressions. A value of zero means always break if-else \
                expressions.";
            single_line_let_else_max_width: SingleLineLetElseMaxWidth, false, "Maximum line length \
                for single line let-else statements. A value of zero means always format the \
                divergent `else` block over multiple lines.";

            // Options that are used by the tests
            stable_option: StableOption, true, "A stable option";
            unstable_option: UnstableOption, false, "An unstable option";
            partially_unstable_option: PartiallyUnstable, true, "A partially unstable option";
            edition: EditionConfig, true, "blah";
            style_edition: StyleEditionConfig, true, "blah";
            version: VersionConfig, false, "blah blah"
        }

        #[cfg(test)]
        mod partially_unstable_option {
            use super::{Config, PartialConfig, PartiallyUnstableOption};
            use rustfmt_config_proc_macro::{nightly_only_test, stable_only_test};
            use std::path::Path;

            /// From the config file, we can fill with a stable variant
            #[test]
            fn test_from_toml_stable_value() {
                let toml = r#"
                    partially_unstable_option = "V2"
                "#;
                let partial_config: PartialConfig = toml::from_str(toml).unwrap();
                let config = Config::default();
                let config = config.fill_from_parsed_config(partial_config, Path::new(""));
                assert_eq!(
                    config.partially_unstable_option(),
                    PartiallyUnstableOption::V2
                );
            }

            /// From the config file, we cannot fill with an unstable variant (stable only)
            #[stable_only_test]
            #[test]
            fn test_from_toml_unstable_value_on_stable() {
                let toml = r#"
                    partially_unstable_option = "V3"
                "#;
                let partial_config: PartialConfig = toml::from_str(toml).unwrap();
                let config = Config::default();
                let config = config.fill_from_parsed_config(partial_config, Path::new(""));
                assert_eq!(
                    config.partially_unstable_option(),
                    // default value from config, i.e. fill failed
                    PartiallyUnstableOption::V1
                );
            }

            /// From the config file, we can fill with an unstable variant (nightly only)
            #[nightly_only_test]
            #[test]
            fn test_from_toml_unstable_value_on_nightly() {
                let toml = r#"
                    partially_unstable_option = "V3"
                "#;
                let partial_config: PartialConfig = toml::from_str(toml).unwrap();
                let config = Config::default();
                let config = config.fill_from_parsed_config(partial_config, Path::new(""));
                assert_eq!(
                    config.partially_unstable_option(),
                    PartiallyUnstableOption::V3
                );
            }
        }
    }

    #[test]
    fn test_config_set() {
        let mut config = Config::default();
        config.set().verbose(Verbosity::Quiet);
        assert_eq!(config.verbose(), Verbosity::Quiet);
        config.set().verbose(Verbosity::Normal);
        assert_eq!(config.verbose(), Verbosity::Normal);
    }

    #[test]
    fn test_config_used_to_toml() {
        let config = Config::default();

        let merge_derives = config.merge_derives();
        let skip_children = config.skip_children();

        let used_options = config.used_options();
        let toml = used_options.to_toml().unwrap();
        assert_eq!(
            toml,
            format!("merge_derives = {merge_derives}\nskip_children = {skip_children}\n",)
        );
    }

    #[test]
    fn test_was_set() {
        let config = Config::from_toml("hard_tabs = true", Path::new("./rustfmt.toml")).unwrap();

        assert_eq!(config.was_set().hard_tabs(), true);
        assert_eq!(config.was_set().verbose(), false);
    }

    const PRINT_DOCS_STABLE_OPTION: &str = "stable_option <boolean> Default: false";
    const PRINT_DOCS_UNSTABLE_OPTION: &str = "unstable_option <boolean> Default: false (unstable)";
    const PRINT_DOCS_PARTIALLY_UNSTABLE_OPTION: &str =
        "partially_unstable_option [V1|V2|V3 (unstable)] Default: V1";

    #[test]
    fn test_print_docs_exclude_unstable() {
        use self::mock::Config;

        let mut output = Vec::new();
        Config::print_docs(&mut output, false);

        let s = str::from_utf8(&output).unwrap();
        assert_eq!(s.contains(PRINT_DOCS_STABLE_OPTION), true);
        assert_eq!(s.contains(PRINT_DOCS_UNSTABLE_OPTION), false);
        assert_eq!(s.contains(PRINT_DOCS_PARTIALLY_UNSTABLE_OPTION), true);
    }

    #[test]
    fn test_print_docs_include_unstable() {
        use self::mock::Config;

        let mut output = Vec::new();
        Config::print_docs(&mut output, true);

        let s = str::from_utf8(&output).unwrap();
        assert_eq!(s.contains(PRINT_DOCS_STABLE_OPTION), true);
        assert_eq!(s.contains(PRINT_DOCS_UNSTABLE_OPTION), true);
        assert_eq!(s.contains(PRINT_DOCS_PARTIALLY_UNSTABLE_OPTION), true);
    }

    #[test]
    fn test_dump_default_config() {
        let default_config = format!(
            r#"max_width = 100
hard_tabs = false
tab_spaces = 4
newline_style = "Auto"
indent_style = "Block"
use_small_heuristics = "Default"
fn_call_width = 60
attr_fn_like_width = 70
struct_lit_width = 18
struct_variant_width = 35
array_width = 60
chain_width = 60
single_line_if_else_max_width = 50
single_line_let_else_max_width = 50
wrap_comments = false
format_code_in_doc_comments = false
doc_comment_code_block_width = 100
comment_width = 80
normalize_comments = false
normalize_doc_attributes = false
format_strings = false
format_macro_matchers = false
format_macro_bodies = true
skip_macro_invocations = []
hex_literal_case = "Preserve"
empty_item_single_line = true
struct_lit_single_line = true
fn_single_line = false
where_single_line = false
imports_indent = "Block"
imports_layout = "Mixed"
imports_granularity = "Preserve"
group_imports = "Preserve"
reorder_imports = true
reorder_modules = true
reorder_impl_items = false
type_punctuation_density = "Wide"
space_before_colon = false
space_after_colon = true
spaces_around_ranges = false
binop_separator = "Front"
remove_nested_parens = true
combine_control_expr = true
short_array_element_width_threshold = 10
overflow_delimited_expr = false
struct_field_align_threshold = 0
enum_discrim_align_threshold = 0
match_arm_blocks = true
match_arm_leading_pipes = "Never"
force_multiline_blocks = false
fn_params_layout = "Tall"
brace_style = "SameLineWhere"
control_brace_style = "AlwaysSameLine"
trailing_semicolon = true
trailing_comma = "Vertical"
match_block_trailing_comma = false
blank_lines_upper_bound = 1
blank_lines_lower_bound = 0
edition = "2015"
style_edition = "2015"
version = "One"
inline_attribute_width = 0
format_generated_files = true
generated_marker_line_search_limit = 5
merge_derives = true
use_try_shorthand = false
use_field_init_shorthand = false
force_explicit_abi = true
condense_wildcard_suffixes = false
color = "Auto"
required_version = "{}"
unstable_features = false
disable_all_formatting = false
skip_children = false
show_parse_errors = true
error_on_line_overflow = false
error_on_unformatted = false
ignore = []
emit_mode = "Files"
make_backup = false
"#,
            env!("CARGO_PKG_VERSION")
        );
        let toml = Config::default().all_options().to_toml().unwrap();
        assert_eq!(&toml, &default_config);
    }

    #[test]
    fn test_dump_style_edition_2024_config() {
        let edition_2024_config = format!(
            r#"max_width = 100
hard_tabs = false
tab_spaces = 4
newline_style = "Auto"
indent_style = "Block"
use_small_heuristics = "Default"
fn_call_width = 60
attr_fn_like_width = 70
struct_lit_width = 18
struct_variant_width = 35
array_width = 60
chain_width = 60
single_line_if_else_max_width = 50
single_line_let_else_max_width = 50
wrap_comments = false
format_code_in_doc_comments = false
doc_comment_code_block_width = 100
comment_width = 80
normalize_comments = false
normalize_doc_attributes = false
format_strings = false
format_macro_matchers = false
format_macro_bodies = true
skip_macro_invocations = []
hex_literal_case = "Preserve"
empty_item_single_line = true
struct_lit_single_line = true
fn_single_line = false
where_single_line = false
imports_indent = "Block"
imports_layout = "Mixed"
imports_granularity = "Preserve"
group_imports = "Preserve"
reorder_imports = true
reorder_modules = true
reorder_impl_items = false
type_punctuation_density = "Wide"
space_before_colon = false
space_after_colon = true
spaces_around_ranges = false
binop_separator = "Front"
remove_nested_parens = true
combine_control_expr = true
short_array_element_width_threshold = 10
overflow_delimited_expr = false
struct_field_align_threshold = 0
enum_discrim_align_threshold = 0
match_arm_blocks = true
match_arm_leading_pipes = "Never"
force_multiline_blocks = false
fn_params_layout = "Tall"
brace_style = "SameLineWhere"
control_brace_style = "AlwaysSameLine"
trailing_semicolon = true
trailing_comma = "Vertical"
match_block_trailing_comma = false
blank_lines_upper_bound = 1
blank_lines_lower_bound = 0
edition = "2015"
style_edition = "2024"
version = "Two"
inline_attribute_width = 0
format_generated_files = true
generated_marker_line_search_limit = 5
merge_derives = true
use_try_shorthand = false
use_field_init_shorthand = false
force_explicit_abi = true
condense_wildcard_suffixes = false
color = "Auto"
required_version = "{}"
unstable_features = false
disable_all_formatting = false
skip_children = false
show_parse_errors = true
error_on_line_overflow = false
error_on_unformatted = false
ignore = []
emit_mode = "Files"
make_backup = false
"#,
            env!("CARGO_PKG_VERSION")
        );
        let toml = Config::default_with_style_edition(StyleEdition::Edition2024)
            .all_options()
            .to_toml()
            .unwrap();
        assert_eq!(&toml, &edition_2024_config);
    }

    #[test]
    fn test_editions_2015_2018_2021_identical() {
        let get_edition_toml = |style_edition: StyleEdition| {
            Config::default_with_style_edition(style_edition)
                .all_options()
                .to_toml()
                .unwrap()
        };
        let edition2015 = get_edition_toml(StyleEdition::Edition2015);
        let edition2018 = get_edition_toml(StyleEdition::Edition2018);
        let edition2021 = get_edition_toml(StyleEdition::Edition2021);
        assert_eq!(edition2015, edition2018);
        assert_eq!(edition2018, edition2021);
    }

    #[stable_only_test]
    #[test]
    fn test_as_not_nightly_channel() {
        let mut config = Config::default();
        assert_eq!(config.was_set().unstable_features(), false);
        config.set().unstable_features(true);
        assert_eq!(config.was_set().unstable_features(), false);
    }

    #[nightly_only_test]
    #[test]
    fn test_as_nightly_channel() {
        let mut config = Config::default();
        config.set().unstable_features(true);
        // When we don't set the config from toml or command line options it
        // doesn't get marked as set by the user.
        assert_eq!(config.was_set().unstable_features(), false);
        config.set().unstable_features(true);
        assert_eq!(config.unstable_features(), true);
    }

    #[nightly_only_test]
    #[test]
    fn test_unstable_from_toml() {
        let config =
            Config::from_toml("unstable_features = true", Path::new("./rustfmt.toml")).unwrap();
        assert_eq!(config.was_set().unstable_features(), true);
        assert_eq!(config.unstable_features(), true);
    }

    #[test]
    fn test_set_cli() {
        let mut config = Config::default();
        assert_eq!(config.was_set().edition(), false);
        assert_eq!(config.was_set_cli().edition(), false);
        config.set().edition(Edition::Edition2021);
        assert_eq!(config.was_set().edition(), false);
        assert_eq!(config.was_set_cli().edition(), false);
        config.set_cli().edition(Edition::Edition2021);
        assert_eq!(config.was_set().edition(), false);
        assert_eq!(config.was_set_cli().edition(), true);
        assert_eq!(config.was_set_cli().emit_mode(), false);
    }

    #[cfg(test)]
    mod deprecated_option_merge_imports {
        use super::*;

        #[nightly_only_test]
        #[test]
        fn test_old_option_set() {
            let toml = r#"
                unstable_features = true
                merge_imports = true
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.imports_granularity(), ImportGranularity::Crate);
        }

        #[nightly_only_test]
        #[test]
        fn test_both_set() {
            let toml = r#"
                unstable_features = true
                merge_imports = true
                imports_granularity = "Preserve"
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.imports_granularity(), ImportGranularity::Preserve);
        }

        #[nightly_only_test]
        #[test]
        fn test_new_overridden() {
            let toml = r#"
                unstable_features = true
                merge_imports = true
            "#;
            let mut config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            config.override_value("imports_granularity", "Preserve");
            assert_eq!(config.imports_granularity(), ImportGranularity::Preserve);
        }

        #[nightly_only_test]
        #[test]
        fn test_old_overridden() {
            let toml = r#"
                unstable_features = true
                imports_granularity = "Module"
            "#;
            let mut config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            config.override_value("merge_imports", "true");
            // no effect: the new option always takes precedence
            assert_eq!(config.imports_granularity(), ImportGranularity::Module);
        }
    }

    #[cfg(test)]
    mod use_small_heuristics {
        use super::*;

        #[test]
        fn test_default_sets_correct_widths() {
            let toml = r#"
                use_small_heuristics = "Default"
                max_width = 200
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.array_width(), 120);
            assert_eq!(config.attr_fn_like_width(), 140);
            assert_eq!(config.chain_width(), 120);
            assert_eq!(config.fn_call_width(), 120);
            assert_eq!(config.single_line_if_else_max_width(), 100);
            assert_eq!(config.struct_lit_width(), 36);
            assert_eq!(config.struct_variant_width(), 70);
        }

        #[test]
        fn test_max_sets_correct_widths() {
            let toml = r#"
                use_small_heuristics = "Max"
                max_width = 120
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.array_width(), 120);
            assert_eq!(config.attr_fn_like_width(), 120);
            assert_eq!(config.chain_width(), 120);
            assert_eq!(config.fn_call_width(), 120);
            assert_eq!(config.single_line_if_else_max_width(), 120);
            assert_eq!(config.struct_lit_width(), 120);
            assert_eq!(config.struct_variant_width(), 120);
        }

        #[test]
        fn test_off_sets_correct_widths() {
            let toml = r#"
                use_small_heuristics = "Off"
                max_width = 100
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.array_width(), usize::MAX);
            assert_eq!(config.attr_fn_like_width(), usize::MAX);
            assert_eq!(config.chain_width(), usize::MAX);
            assert_eq!(config.fn_call_width(), usize::MAX);
            assert_eq!(config.single_line_if_else_max_width(), 0);
            assert_eq!(config.struct_lit_width(), 0);
            assert_eq!(config.struct_variant_width(), 0);
        }

        #[test]
        fn test_override_works_with_default() {
            let toml = r#"
                use_small_heuristics = "Default"
                array_width = 20
                attr_fn_like_width = 40
                chain_width = 20
                fn_call_width = 90
                single_line_if_else_max_width = 40
                struct_lit_width = 30
                struct_variant_width = 34
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.array_width(), 20);
            assert_eq!(config.attr_fn_like_width(), 40);
            assert_eq!(config.chain_width(), 20);
            assert_eq!(config.fn_call_width(), 90);
            assert_eq!(config.single_line_if_else_max_width(), 40);
            assert_eq!(config.struct_lit_width(), 30);
            assert_eq!(config.struct_variant_width(), 34);
        }

        #[test]
        fn test_override_with_max() {
            let toml = r#"
                use_small_heuristics = "Max"
                array_width = 20
                attr_fn_like_width = 40
                chain_width = 20
                fn_call_width = 90
                single_line_if_else_max_width = 40
                struct_lit_width = 30
                struct_variant_width = 34
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.array_width(), 20);
            assert_eq!(config.attr_fn_like_width(), 40);
            assert_eq!(config.chain_width(), 20);
            assert_eq!(config.fn_call_width(), 90);
            assert_eq!(config.single_line_if_else_max_width(), 40);
            assert_eq!(config.struct_lit_width(), 30);
            assert_eq!(config.struct_variant_width(), 34);
        }

        #[test]
        fn test_override_with_off() {
            let toml = r#"
                use_small_heuristics = "Off"
                array_width = 20
                attr_fn_like_width = 40
                chain_width = 20
                fn_call_width = 90
                single_line_if_else_max_width = 40
                struct_lit_width = 30
                struct_variant_width = 34
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.array_width(), 20);
            assert_eq!(config.attr_fn_like_width(), 40);
            assert_eq!(config.chain_width(), 20);
            assert_eq!(config.fn_call_width(), 90);
            assert_eq!(config.single_line_if_else_max_width(), 40);
            assert_eq!(config.struct_lit_width(), 30);
            assert_eq!(config.struct_variant_width(), 34);
        }

        #[test]
        fn test_fn_call_width_config_exceeds_max_width() {
            let toml = r#"
                max_width = 90
                fn_call_width = 95
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.fn_call_width(), 90);
        }

        #[test]
        fn test_attr_fn_like_width_config_exceeds_max_width() {
            let toml = r#"
                max_width = 80
                attr_fn_like_width = 90
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.attr_fn_like_width(), 80);
        }

        #[test]
        fn test_struct_lit_config_exceeds_max_width() {
            let toml = r#"
                max_width = 78
                struct_lit_width = 90
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.struct_lit_width(), 78);
        }

        #[test]
        fn test_struct_variant_width_config_exceeds_max_width() {
            let toml = r#"
                max_width = 80
                struct_variant_width = 90
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.struct_variant_width(), 80);
        }

        #[test]
        fn test_array_width_config_exceeds_max_width() {
            let toml = r#"
                max_width = 60
                array_width = 80
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.array_width(), 60);
        }

        #[test]
        fn test_chain_width_config_exceeds_max_width() {
            let toml = r#"
                max_width = 80
                chain_width = 90
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.chain_width(), 80);
        }

        #[test]
        fn test_single_line_if_else_max_width_config_exceeds_max_width() {
            let toml = r#"
                max_width = 70
                single_line_if_else_max_width = 90
            "#;
            let config = Config::from_toml(toml, Path::new("./rustfmt.toml")).unwrap();
            assert_eq!(config.single_line_if_else_max_width(), 70);
        }

        #[test]
        fn test_override_fn_call_width_exceeds_max_width() {
            let mut config = Config::default();
            config.override_value("fn_call_width", "101");
            assert_eq!(config.fn_call_width(), 100);
        }

        #[test]
        fn test_override_attr_fn_like_width_exceeds_max_width() {
            let mut config = Config::default();
            config.override_value("attr_fn_like_width", "101");
            assert_eq!(config.attr_fn_like_width(), 100);
        }

        #[test]
        fn test_override_struct_lit_exceeds_max_width() {
            let mut config = Config::default();
            config.override_value("struct_lit_width", "101");
            assert_eq!(config.struct_lit_width(), 100);
        }

        #[test]
        fn test_override_struct_variant_width_exceeds_max_width() {
            let mut config = Config::default();
            config.override_value("struct_variant_width", "101");
            assert_eq!(config.struct_variant_width(), 100);
        }

        #[test]
        fn test_override_array_width_exceeds_max_width() {
            let mut config = Config::default();
            config.override_value("array_width", "101");
            assert_eq!(config.array_width(), 100);
        }

        #[test]
        fn test_override_chain_width_exceeds_max_width() {
            let mut config = Config::default();
            config.override_value("chain_width", "101");
            assert_eq!(config.chain_width(), 100);
        }

        #[test]
        fn test_override_single_line_if_else_max_width_exceeds_max_width() {
            let mut config = Config::default();
            config.override_value("single_line_if_else_max_width", "101");
            assert_eq!(config.single_line_if_else_max_width(), 100);
        }
    }

    #[cfg(test)]
    mod partially_unstable_option {
        use super::mock::{Config, PartiallyUnstableOption};

        /// From the command line, we can override with a stable variant.
        #[test]
        fn test_override_stable_value() {
            let mut config = Config::default();
            config.override_value("partially_unstable_option", "V2");
            assert_eq!(
                config.partially_unstable_option(),
                PartiallyUnstableOption::V2
            );
        }

        /// From the command line, we can override with an unstable variant.
        #[test]
        fn test_override_unstable_value() {
            let mut config = Config::default();
            config.override_value("partially_unstable_option", "V3");
            assert_eq!(
                config.partially_unstable_option(),
                PartiallyUnstableOption::V3
            );
        }
    }

    #[test]
    fn test_override_skip_macro_invocations() {
        let mut config = Config::default();
        config.override_value("skip_macro_invocations", r#"["*", "println"]"#);
        assert_eq!(
            config.skip_macro_invocations(),
            MacroSelectors(vec![
                MacroSelector::All,
                MacroSelector::Name(MacroName::new("println".to_owned()))
            ])
        );
    }
}
