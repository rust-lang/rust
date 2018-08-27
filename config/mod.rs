// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use regex::Regex;
use std::cell::Cell;
use std::default::Default;
use std::fs::File;
use std::io::{Error, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::{env, fs};

use config::config_type::ConfigType;
pub use config::file_lines::{FileLines, FileName, Range};
pub use config::lists::*;
pub use config::options::*;

#[macro_use]
pub mod config_type;
#[macro_use]
pub mod options;

pub mod file_lines;
pub mod license;
pub mod lists;

/// This macro defines configuration options used in rustfmt. Each option
/// is defined as follows:
///
/// `name: value type, default value, is stable, description;`
create_config! {
    // Fundamental stuff
    max_width: usize, 100, true, "Maximum width of each line";
    hard_tabs: bool, false, true, "Use tab characters for indentation, spaces for alignment";
    tab_spaces: usize, 4, true, "Number of spaces per tab";
    newline_style: NewlineStyle, NewlineStyle::Auto, true, "Unix or Windows line endings";
    use_small_heuristics: Heuristics, Heuristics::Default, true, "Whether to use different \
        formatting for items and expressions if they satisfy a heuristic notion of 'small'";
    indent_style: IndentStyle, IndentStyle::Block, false, "How do we indent expressions or items";

    // Comments. macros, and strings
    wrap_comments: bool, false, false, "Break comments to fit on the line";
    comment_width: usize, 80, false,
        "Maximum length of comments. No effect unless wrap_comments = true";
    normalize_comments: bool, false, false, "Convert /* */ comments to // comments where possible";
    license_template_path: String, String::default(), false,
        "Beginning of file must match license template";
    format_strings: bool, false, false, "Format string literals where necessary";
    format_macro_matchers: bool, false, false,
        "Format the metavariable matching patterns in macros";
    format_macro_bodies: bool, true, false, "Format the bodies of macros";

    // Single line expressions and items
    empty_item_single_line: bool, true, false,
        "Put empty-body functions and impls on a single line";
    struct_lit_single_line: bool, true, false,
        "Put small struct literals on a single line";
    fn_single_line: bool, false, false, "Put single-expression functions on a single line";
    where_single_line: bool, false, false, "Force where clauses to be on a single line";

    // Imports
    imports_indent: IndentStyle, IndentStyle::Block, false, "Indent of imports";
    imports_layout: ListTactic, ListTactic::Mixed, false, "Item layout inside a import block";
    merge_imports: bool, false, false, "Merge imports";

    // Ordering
    reorder_imports: bool, true, true, "Reorder import and extern crate statements alphabetically";
    reorder_modules: bool, true, true, "Reorder module statements alphabetically in group";
    reorder_impl_items: bool, false, false, "Reorder impl items";

    // Spaces around punctuation
    type_punctuation_density: TypeDensity, TypeDensity::Wide, false,
        "Determines if '+' or '=' are wrapped in spaces in the punctuation of types";
    space_before_colon: bool, false, false, "Leave a space before the colon";
    space_after_colon: bool, true, false, "Leave a space after the colon";
    spaces_around_ranges: bool, false, false, "Put spaces around the  .. and ..= range operators";
    binop_separator: SeparatorPlace, SeparatorPlace::Front, false,
        "Where to put a binary operator when a binary expression goes multiline";

    // Misc.
    remove_nested_parens: bool, true, true, "Remove nested parens";
    combine_control_expr: bool, true, false, "Combine control expressions with function calls";
    struct_field_align_threshold: usize, 0, false, "Align struct fields if their diffs fits within \
                                             threshold";
    match_arm_blocks: bool, true, false, "Wrap the body of arms in blocks when it does not fit on \
        the same line with the pattern of arms";
    force_multiline_blocks: bool, false, false,
        "Force multiline closure bodies and match arms to be wrapped in a block";
    fn_args_density: Density, Density::Tall, false, "Argument density in functions";
    brace_style: BraceStyle, BraceStyle::SameLineWhere, false, "Brace style for items";
    control_brace_style: ControlBraceStyle, ControlBraceStyle::AlwaysSameLine, false,
        "Brace style for control flow constructs";
    trailing_semicolon: bool, true, false,
        "Add trailing semicolon after break, continue and return";
    trailing_comma: SeparatorTactic, SeparatorTactic::Vertical, false,
        "How to handle trailing commas for lists";
    match_block_trailing_comma: bool, false, false,
        "Put a trailing comma after a block based match arm (non-block arms are not affected)";
    blank_lines_upper_bound: usize, 1, false,
        "Maximum number of blank lines which can be put between items";
    blank_lines_lower_bound: usize, 0, false,
        "Minimum number of blank lines which must be put between items";
    edition: Edition, Edition::Edition2015, false, "The edition of the parser (RFC 2052)";

    // Options that can change the source code beyond whitespace/blocks (somewhat linty things)
    merge_derives: bool, true, true, "Merge multiple `#[derive(...)]` into a single one";
    use_try_shorthand: bool, false, true, "Replace uses of the try! macro by the ? shorthand";
    use_field_init_shorthand: bool, false, true, "Use field initialization shorthand if possible";
    force_explicit_abi: bool, true, true, "Always print the abi for extern items";
    condense_wildcard_suffixes: bool, false, false, "Replace strings of _ wildcards by a single .. \
                                                     in tuple patterns";

    // Control options (changes the operation of rustfmt, rather than the formatting)
    color: Color, Color::Auto, false,
        "What Color option to use when none is supplied: Always, Never, Auto";
    required_version: String, env!("CARGO_PKG_VERSION").to_owned(), false,
        "Require a specific version of rustfmt";
    unstable_features: bool, false, false,
            "Enables unstable features. Only available on nightly channel";
    disable_all_formatting: bool, false, false, "Don't reformat anything";
    skip_children: bool, false, false, "Don't reformat out of line modules";
    hide_parse_errors: bool, false, false, "Hide errors from the parser";
    error_on_line_overflow: bool, false, false, "Error if unable to get all lines within max_width";
    error_on_unformatted: bool, false, false,
        "Error if unable to get comments or string literals within max_width, \
         or they are left with trailing whitespaces";
    report_todo: ReportTactic, ReportTactic::Never, false,
        "Report all, none or unnumbered occurrences of TODO in source file comments";
    report_fixme: ReportTactic, ReportTactic::Never, false,
        "Report all, none or unnumbered occurrences of FIXME in source file comments";
    ignore: IgnoreList, IgnoreList::default(), false,
        "Skip formatting the specified files and directories";

    // Not user-facing
    verbose: Verbosity, Verbosity::Normal, false, "How much to information to emit to the user";
    file_lines: FileLines, FileLines::all(), false,
        "Lines to format; this is not supported in rustfmt.toml, and can only be specified \
         via the --file-lines option";
    width_heuristics: WidthHeuristics, WidthHeuristics::scaled(100), false,
        "'small' heuristic values";
    emit_mode: EmitMode, EmitMode::Files, false,
        "What emit Mode to use when none is supplied";
    make_backup: bool, false, false, "Backup changed files";
}

/// Load a config by checking the client-supplied options and if appropriate, the
/// file system (including searching the file system for overrides).
pub fn load_config<O: CliOptions>(
    file_path: Option<&Path>,
    options: Option<O>,
) -> Result<(Config, Option<PathBuf>), Error> {
    let over_ride = match options {
        Some(ref opts) => config_path(opts)?,
        None => None,
    };

    let result = if let Some(over_ride) = over_ride {
        Config::from_toml_path(over_ride.as_ref()).map(|p| (p, Some(over_ride.to_owned())))
    } else if let Some(file_path) = file_path {
        Config::from_resolved_toml_path(file_path)
    } else {
        Ok((Config::default(), None))
    };

    result.map(|(mut c, p)| {
        if let Some(options) = options {
            options.apply_to(&mut c);
        }
        (c, p)
    })
}

// Check for the presence of known config file names (`rustfmt.toml, `.rustfmt.toml`) in `dir`
//
// Return the path if a config file exists, empty if no file exists, and Error for IO errors
fn get_toml_path(dir: &Path) -> Result<Option<PathBuf>, Error> {
    const CONFIG_FILE_NAMES: [&str; 2] = [".rustfmt.toml", "rustfmt.toml"];
    for config_file_name in &CONFIG_FILE_NAMES {
        let config_file = dir.join(config_file_name);
        match fs::metadata(&config_file) {
            // Only return if it's a file to handle the unlikely situation of a directory named
            // `rustfmt.toml`.
            Ok(ref md) if md.is_file() => return Ok(Some(config_file)),
            // Return the error if it's something other than `NotFound`; otherwise we didn't
            // find the project file yet, and continue searching.
            Err(e) => {
                if e.kind() != ErrorKind::NotFound {
                    return Err(e);
                }
            }
            _ => {}
        }
    }
    Ok(None)
}

fn config_path(options: &CliOptions) -> Result<Option<PathBuf>, Error> {
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
        path => Ok(path.map(|p| p.to_owned())),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::str;

    #[allow(dead_code)]
    mod mock {
        use super::super::*;

        create_config! {
            // Options that are used by the generated functions
            max_width: usize, 100, true, "Maximum width of each line";
            use_small_heuristics: Heuristics, Heuristics::Default, true,
                "Whether to use different formatting for items and \
                 expressions if they satisfy a heuristic notion of 'small'.";
            license_template_path: String, String::default(), false,
                "Beginning of file must match license template";
            required_version: String, env!("CARGO_PKG_VERSION").to_owned(), false,
                "Require a specific version of rustfmt.";
            ignore: IgnoreList, IgnoreList::default(), false,
                "Skip formatting the specified files and directories.";
            verbose: Verbosity, Verbosity::Normal, false,
                "How much to information to emit to the user";
            file_lines: FileLines, FileLines::all(), false,
                "Lines to format; this is not supported in rustfmt.toml, and can only be specified \
                    via the --file-lines option";
            width_heuristics: WidthHeuristics, WidthHeuristics::scaled(100), false,
                "'small' heuristic values";

            // Options that are used by the tests
            stable_option: bool, false, true, "A stable option";
            unstable_option: bool, false, false, "An unstable option";
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
            format!(
                "merge_derives = {}\nskip_children = {}\n",
                merge_derives, skip_children,
            )
        );
    }

    #[test]
    fn test_was_set() {
        use std::path::Path;
        let config = Config::from_toml("hard_tabs = true", Path::new("")).unwrap();

        assert_eq!(config.was_set().hard_tabs(), true);
        assert_eq!(config.was_set().verbose(), false);
    }

    #[test]
    fn test_print_docs_exclude_unstable() {
        use self::mock::Config;

        let mut output = Vec::new();
        Config::print_docs(&mut output, false);

        let s = str::from_utf8(&output).unwrap();

        assert_eq!(s.contains("stable_option"), true);
        assert_eq!(s.contains("unstable_option"), false);
        assert_eq!(s.contains("(unstable)"), false);
    }

    #[test]
    fn test_print_docs_include_unstable() {
        use self::mock::Config;

        let mut output = Vec::new();
        Config::print_docs(&mut output, true);

        let s = str::from_utf8(&output).unwrap();
        assert_eq!(s.contains("stable_option"), true);
        assert_eq!(s.contains("unstable_option"), true);
        assert_eq!(s.contains("(unstable)"), true);
    }

    // FIXME(#2183) these tests cannot be run in parallel because they use env vars
    // #[test]
    // fn test_as_not_nightly_channel() {
    //     let mut config = Config::default();
    //     assert_eq!(config.was_set().unstable_features(), false);
    //     config.set().unstable_features(true);
    //     assert_eq!(config.was_set().unstable_features(), false);
    // }

    // #[test]
    // fn test_as_nightly_channel() {
    //     let v = ::std::env::var("CFG_RELEASE_CHANNEL").unwrap_or(String::from(""));
    //     ::std::env::set_var("CFG_RELEASE_CHANNEL", "nightly");
    //     let mut config = Config::default();
    //     config.set().unstable_features(true);
    //     assert_eq!(config.was_set().unstable_features(), false);
    //     config.set().unstable_features(true);
    //     assert_eq!(config.unstable_features(), true);
    //     ::std::env::set_var("CFG_RELEASE_CHANNEL", v);
    // }

    // #[test]
    // fn test_unstable_from_toml() {
    //     let mut config = Config::from_toml("unstable_features = true").unwrap();
    //     assert_eq!(config.was_set().unstable_features(), false);
    //     let v = ::std::env::var("CFG_RELEASE_CHANNEL").unwrap_or(String::from(""));
    //     ::std::env::set_var("CFG_RELEASE_CHANNEL", "nightly");
    //     config = Config::from_toml("unstable_features = true").unwrap();
    //     assert_eq!(config.was_set().unstable_features(), true);
    //     assert_eq!(config.unstable_features(), true);
    //     ::std::env::set_var("CFG_RELEASE_CHANNEL", v);
    // }
}
