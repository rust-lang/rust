// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate syntax;
extern crate toml;

use std::{env, fs};
use std::cell::Cell;
use std::default::Default;
use std::fs::File;
use std::io::{Error, ErrorKind, Read};
use std::path::{Path, PathBuf};

#[macro_use]
mod config_type;
#[macro_use]
mod options;

pub mod file_lines;
pub mod lists;
pub mod summary;

use config_type::ConfigType;
use file_lines::FileLines;
pub use lists::*;
pub use options::*;
use summary::Summary;

/// This macro defines configuration options used in rustfmt. Each option
/// is defined as follows:
///
/// `name: value type, default value, is stable, description;`
create_config! {
    // Fundamental stuff
    max_width: usize, 100, true, "Maximum width of each line";
    hard_tabs: bool, false, true, "Use tab characters for indentation, spaces for alignment";
    tab_spaces: usize, 4, true, "Number of spaces per tab";
    newline_style: NewlineStyle, NewlineStyle::Unix, true, "Unix or Windows line endings";
    indent_style: IndentStyle, IndentStyle::Block, false, "How do we indent expressions or items.";
    use_small_heuristics: bool, true, false, "Whether to use different formatting for items and \
        expressions if they satisfy a heuristic notion of 'small'.";

    // strings and comments
    format_strings: bool, false, false, "Format string literals where necessary";
    wrap_comments: bool, false, true, "Break comments to fit on the line";
    comment_width: usize, 80, false,
        "Maximum length of comments. No effect unless wrap_comments = true";
    normalize_comments: bool, false, true, "Convert /* */ comments to // comments where possible";

    // Single line expressions and items.
    empty_item_single_line: bool, true, false,
        "Put empty-body functions and impls on a single line";
    struct_lit_single_line: bool, true, false,
        "Put small struct literals on a single line";
    fn_single_line: bool, false, false, "Put single-expression functions on a single line";
    where_single_line: bool, false, false, "To force single line where layout";

    // Imports
    imports_indent: IndentStyle, IndentStyle::Visual, false, "Indent of imports";
    imports_layout: ListTactic, ListTactic::Mixed, false, "Item layout inside a import block";

    // Ordering
    reorder_extern_crates: bool, true, false, "Reorder extern crate statements alphabetically";
    reorder_extern_crates_in_group: bool, true, false, "Reorder extern crate statements in group";
    reorder_imports: bool, false, false, "Reorder import statements alphabetically";
    reorder_imports_in_group: bool, false, false, "Reorder import statements in group";
    reorder_imported_names: bool, true, false,
        "Reorder lists of names in import statements alphabetically";
    reorder_modules: bool, false, false, "Reorder module statemtents alphabetically in group";

    // Spaces around punctuation
    binop_separator: SeparatorPlace, SeparatorPlace::Front, false,
        "Where to put a binary operator when a binary expression goes multiline.";
    type_punctuation_density: TypeDensity, TypeDensity::Wide, false,
        "Determines if '+' or '=' are wrapped in spaces in the punctuation of types";
    space_before_colon: bool, false, false, "Leave a space before the colon";
    space_after_colon: bool, true, false, "Leave a space after the colon";
    spaces_around_ranges: bool, false, false, "Put spaces around the  .. and ... range operators";
    spaces_within_parens_and_brackets: bool, false, false,
        "Put spaces within non-empty parentheses or brackets";

    // Misc.
    combine_control_expr: bool, true, false, "Combine control expressions with function calls.";
    struct_field_align_threshold: usize, 0, false, "Align struct fields if their diffs fits within \
                                             threshold.";
    remove_blank_lines_at_start_or_end_of_block: bool, true, false,
        "Remove blank lines at start or end of a block";
    match_arm_blocks: bool, true, false, "Wrap the body of arms in blocks when it does not fit on \
        the same line with the pattern of arms";
    force_multiline_blocks: bool, false, false,
        "Force multiline closure bodies and match arms to be wrapped in a block";
    fn_args_density: Density, Density::Tall, false, "Argument density in functions";
    brace_style: BraceStyle, BraceStyle::SameLineWhere, false, "Brace style for items";
    control_brace_style: ControlBraceStyle, ControlBraceStyle::AlwaysSameLine, false,
        "Brace style for control flow constructs";
    trailing_comma: SeparatorTactic, SeparatorTactic::Vertical, false,
        "How to handle trailing commas for lists";
    trailing_semicolon: bool, true, false,
        "Add trailing semicolon after break, continue and return";
    match_block_trailing_comma: bool, false, false,
        "Put a trailing comma after a block based match arm (non-block arms are not affected)";
    blank_lines_upper_bound: usize, 1, false,
        "Maximum number of blank lines which can be put between items.";
    blank_lines_lower_bound: usize, 0, false,
        "Minimum number of blank lines which must be put between items.";

    // Options that can change the source code beyond whitespace/blocks (somewhat linty things)
    merge_derives: bool, true, true, "Merge multiple `#[derive(...)]` into a single one";
    use_try_shorthand: bool, false, false, "Replace uses of the try! macro by the ? shorthand";
    condense_wildcard_suffixes: bool, false, false, "Replace strings of _ wildcards by a single .. \
                                              in tuple patterns";
    force_explicit_abi: bool, true, true, "Always print the abi for extern items";
    use_field_init_shorthand: bool, false, false, "Use field initialization shorthand if possible";

    // Control options (changes the operation of rustfmt, rather than the formatting)
    write_mode: WriteMode, WriteMode::Overwrite, false,
        "What Write Mode to use when none is supplied: \
         Replace, Overwrite, Display, Plain, Diff, Coverage";
    color: Color, Color::Auto, false,
        "What Color option to use when none is supplied: Always, Never, Auto";
    required_version: String, env!("CARGO_PKG_VERSION").to_owned(), false,
        "Require a specific version of rustfmt.";
    unstable_features: bool, false, true,
            "Enables unstable features. Only available on nightly channel";
    disable_all_formatting: bool, false, false, "Don't reformat anything";
    skip_children: bool, false, false, "Don't reformat out of line modules";
    hide_parse_errors: bool, false, false, "Hide errors from the parser";
    error_on_line_overflow: bool, true, false, "Error if unable to get all lines within max_width";
    error_on_unformatted: bool, false, false,
        "Error if unable to get comments or string literals within max_width, \
         or they are left with trailing whitespaces";
    report_todo: ReportTactic, ReportTactic::Never, false,
        "Report all, none or unnumbered occurrences of TODO in source file comments";
    report_fixme: ReportTactic, ReportTactic::Never, false,
        "Report all, none or unnumbered occurrences of FIXME in source file comments";

    // Not user-facing.
    verbose: bool, false, false, "Use verbose output";
    file_lines: FileLines, FileLines::all(), false,
        "Lines to format; this is not supported in rustfmt.toml, and can only be specified \
         via the --file-lines option";
    width_heuristics: WidthHeuristics, WidthHeuristics::scaled(100), false,
        "'small' heuristic values";
}

/// Check for the presence of known config file names (`rustfmt.toml, `.rustfmt.toml`) in `dir`
///
/// Return the path if a config file exists, empty if no file exists, and Error for IO errors
pub fn get_toml_path(dir: &Path) -> Result<Option<PathBuf>, Error> {
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

#[cfg(test)]
mod test {
    use super::Config;

    #[test]
    fn test_config_set() {
        let mut config = Config::default();
        config.set().verbose(false);
        assert_eq!(config.verbose(), false);
        config.set().verbose(true);
        assert_eq!(config.verbose(), true);
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
        let config = Config::from_toml("hard_tabs = true").unwrap();

        assert_eq!(config.was_set().hard_tabs(), true);
        assert_eq!(config.was_set().verbose(), false);
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
