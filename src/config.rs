// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate toml;

use std::{env, fs};
use std::cell::Cell;
use std::default::Default;
use std::fs::File;
use std::io::{Error, ErrorKind, Read};
use std::path::{Path, PathBuf};

use Summary;
use file_lines::FileLines;
use lists::{ListTactic, SeparatorPlace, SeparatorTactic};

/// Check if we're in a nightly build.
///
/// The environment variable `CFG_RELEASE_CHANNEL` is set during the rustc bootstrap
/// to "stable", "beta", or "nightly" depending on what toolchain is being built.
/// If we are being built as part of the stable or beta toolchains, we want
/// to disable unstable configuration options.
///
/// If we're being built by cargo (e.g. `cargo +nightly install rustfmt-nightly`),
/// `CFG_RELEASE_CHANNEL` is not set. As we only support being built against the
/// nightly compiler when installed from crates.io, default to nightly mode.
macro_rules! is_nightly_channel {
    () => {
        option_env!("CFG_RELEASE_CHANNEL")
            .map(|c| c == "nightly")
            .unwrap_or(true)
    }
}

macro_rules! configuration_option_enum{
    ($e:ident: $( $x:ident ),+ $(,)*) => {
        #[derive(Copy, Clone, Eq, PartialEq, Debug)]
        pub enum $e {
            $( $x ),+
        }

        impl_enum_serialize_and_deserialize!($e, $( $x ),+);
    }
}

configuration_option_enum! { NewlineStyle:
    Windows, // \r\n
    Unix, // \n
    Native, // \r\n in Windows, \n on other platforms
}

configuration_option_enum! { BraceStyle:
    AlwaysNextLine,
    PreferSameLine,
    // Prefer same line except where there is a where clause, in which case force
    // the brace to the next line.
    SameLineWhere,
}

configuration_option_enum! { ControlBraceStyle:
    // K&R style, Rust community default
    AlwaysSameLine,
    // Stroustrup style
    ClosingNextLine,
    // Allman style
    AlwaysNextLine,
}

configuration_option_enum! { IndentStyle:
    // First line on the same line as the opening brace, all lines aligned with
    // the first line.
    Visual,
    // First line is on a new line and all lines align with block indent.
    Block,
}

configuration_option_enum! { Density:
    // Fit as much on one line as possible.
    Compressed,
    // Use more lines.
    Tall,
    // Place every item on a separate line.
    Vertical,
}

configuration_option_enum! { TypeDensity:
    // No spaces around "=" and "+"
    Compressed,
    // Spaces around " = " and " + "
    Wide,
}

impl Density {
    pub fn to_list_tactic(self) -> ListTactic {
        match self {
            Density::Compressed => ListTactic::Mixed,
            Density::Tall => ListTactic::HorizontalVertical,
            Density::Vertical => ListTactic::Vertical,
        }
    }
}

configuration_option_enum! { ReportTactic:
    Always,
    Unnumbered,
    Never,
}

configuration_option_enum! { WriteMode:
    // Backs the original file up and overwrites the original.
    Replace,
    // Overwrites original file without backup.
    Overwrite,
    // Writes the output to stdout.
    Display,
    // Writes the diff to stdout.
    Diff,
    // Displays how much of the input file was processed
    Coverage,
    // Unfancy stdout
    Plain,
    // Outputs a checkstyle XML file.
    Checkstyle,
}

configuration_option_enum! { Color:
    // Always use color, whether it is a piped or terminal output
    Always,
    // Never use color
    Never,
    // Automatically use color, if supported by terminal
    Auto,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct WidthHeuristics {
    // Maximum width of the args of a function call before falling back
    // to vertical formatting.
    pub fn_call_width: usize,
    // Maximum width in the body of a struct lit before falling back to
    // vertical formatting.
    pub struct_lit_width: usize,
    // Maximum width in the body of a struct variant before falling back
    // to vertical formatting.
    pub struct_variant_width: usize,
    // Maximum width of an array literal before falling back to vertical
    // formatting.
    pub array_width: usize,
    // Maximum length of a chain to fit on a single line.
    pub chain_width: usize,
    // Maximum line length for single line if-else expressions. A value
    // of zero means always break if-else expressions.
    pub single_line_if_else_max_width: usize,
}

impl WidthHeuristics {
    // Using this WidthHeuristics means we ignore heuristics.
    fn null() -> WidthHeuristics {
        WidthHeuristics {
            fn_call_width: usize::max_value(),
            struct_lit_width: 0,
            struct_variant_width: 0,
            array_width: usize::max_value(),
            chain_width: usize::max_value(),
            single_line_if_else_max_width: 0,
        }
    }
}

impl Default for WidthHeuristics {
    fn default() -> WidthHeuristics {
        WidthHeuristics {
            fn_call_width: 60,
            struct_lit_width: 18,
            struct_variant_width: 35,
            array_width: 60,
            chain_width: 60,
            single_line_if_else_max_width: 50,
        }
    }
}

impl ::std::str::FromStr for WidthHeuristics {
    type Err = &'static str;

    fn from_str(_: &str) -> Result<Self, Self::Err> {
        Err("WidthHeuristics is not parsable")
    }
}

impl ::config::ConfigType for WidthHeuristics {
    fn doc_hint() -> String {
        String::new()
    }
}

/// Trait for types that can be used in `Config`.
pub trait ConfigType: Sized {
    /// Returns hint text for use in `Config::print_docs()`. For enum types, this is a
    /// pipe-separated list of variants; for other types it returns "<type>".
    fn doc_hint() -> String;
}

impl ConfigType for bool {
    fn doc_hint() -> String {
        String::from("<boolean>")
    }
}

impl ConfigType for usize {
    fn doc_hint() -> String {
        String::from("<unsigned integer>")
    }
}

impl ConfigType for isize {
    fn doc_hint() -> String {
        String::from("<signed integer>")
    }
}

impl ConfigType for String {
    fn doc_hint() -> String {
        String::from("<string>")
    }
}

impl ConfigType for FileLines {
    fn doc_hint() -> String {
        String::from("<json>")
    }
}

pub struct ConfigHelpItem {
    option_name: &'static str,
    doc_string: &'static str,
    variant_names: String,
    default: &'static str,
}

impl ConfigHelpItem {
    pub fn option_name(&self) -> &'static str {
        self.option_name
    }

    pub fn doc_string(&self) -> &'static str {
        self.doc_string
    }

    pub fn variant_names(&self) -> &String {
        &self.variant_names
    }

    pub fn default(&self) -> &'static str {
        self.default
    }
}

macro_rules! create_config {
    ($($i:ident: $ty:ty, $def:expr, $stb:expr, $( $dstring:expr ),+ );+ $(;)*) => (
        #[derive(Clone)]
        pub struct Config {
            // For each config item, we store a bool indicating whether it has
            // been accessed and the value, and a bool whether the option was
            // manually initialised, or taken from the default,
            $($i: (Cell<bool>, bool, $ty, bool)),+
        }

        // Just like the Config struct but with each property wrapped
        // as Option<T>. This is used to parse a rustfmt.toml that doesn't
        // specify all properties of `Config`.
        // We first parse into `PartialConfig`, then create a default `Config`
        // and overwrite the properties with corresponding values from `PartialConfig`.
        #[derive(Deserialize, Serialize, Clone)]
        pub struct PartialConfig {
            $(pub $i: Option<$ty>),+
        }

        impl PartialConfig {
            pub fn to_toml(&self) -> Result<String, String> {
                // Non-user-facing options can't be specified in TOML
                let mut cloned = self.clone();
                cloned.file_lines = None;
                cloned.verbose = None;
                cloned.width_heuristics = None;

                toml::to_string(&cloned)
                    .map_err(|e| format!("Could not output config: {}", e.to_string()))
            }
        }

        // Macro hygiene won't allow us to make `set_$i()` methods on Config
        // for each item, so this struct is used to give the API to set values:
        // `config.get().option(false)`. It's pretty ugly. Consider replacing
        // with `config.set_option(false)` if we ever get a stable/usable
        // `concat_idents!()`.
        pub struct ConfigSetter<'a>(&'a mut Config);

        impl<'a> ConfigSetter<'a> {
            $(
            pub fn $i(&mut self, value: $ty) {
                (self.0).$i.2 = value;
                if stringify!($i) == "use_small_heuristics" {
                    self.0.set_heuristics();
                }
            }
            )+
        }

        // Query each option, returns true if the user set the option, false if
        // a default was used.
        pub struct ConfigWasSet<'a>(&'a Config);

        impl<'a> ConfigWasSet<'a> {
            $(
            pub fn $i(&self) -> bool {
                (self.0).$i.1
            }
            )+
        }

        impl Config {
            pub fn version_meets_requirement(&self, error_summary: &mut Summary) -> bool {
                if self.was_set().required_version() {
                    let version = env!("CARGO_PKG_VERSION");
                    let required_version = self.required_version();
                    if version != required_version {
                        println!(
                            "Error: rustfmt version ({}) doesn't match the required version ({})",
                            version,
                            required_version,
                        );
                        error_summary.add_formatting_error();
                        return false;
                    }
                }

                true
            }

            $(
            pub fn $i(&self) -> $ty {
                self.$i.0.set(true);
                self.$i.2.clone()
            }
            )+

            pub fn set<'a>(&'a mut self) -> ConfigSetter<'a> {
                ConfigSetter(self)
            }

            pub fn was_set<'a>(&'a self) -> ConfigWasSet<'a> {
                ConfigWasSet(self)
            }

            fn fill_from_parsed_config(mut self, parsed: PartialConfig) -> Config {
            $(
                if let Some(val) = parsed.$i {
                    if self.$i.3 {
                        self.$i.1 = true;
                        self.$i.2 = val;
                    } else {
                        if is_nightly_channel!() {
                            self.$i.1 = true;
                            self.$i.2 = val;
                        } else {
                            eprintln!("Warning: can't set `{} = {:?}`, unstable features are only \
                                       available in nightly channel.", stringify!($i), val);
                        }
                    }
                }
            )+
                self.set_heuristics();
                self
            }

            pub fn from_toml(toml: &str) -> Result<Config, String> {
                let parsed: toml::Value =
                    toml.parse().map_err(|e| format!("Could not parse TOML: {}", e))?;
                let mut err: String = String::new();
                {
                    let table = parsed
                        .as_table()
                        .ok_or(String::from("Parsed config was not table"))?;
                    for key in table.keys() {
                        match &**key {
                            $(
                                stringify!($i) => (),
                            )+
                            _ => {
                                let msg =
                                    &format!("Warning: Unknown configuration option `{}`\n", key);
                                err.push_str(msg)
                            }
                        }
                    }
                }
                match parsed.try_into() {
                    Ok(parsed_config) => {
                        if !err.is_empty() {
                            eprint!("{}", err);
                        }
                        Ok(Config::default().fill_from_parsed_config(parsed_config))
                    }
                    Err(e) => {
                        err.push_str("Error: Decoding config file failed:\n");
                        err.push_str(format!("{}\n", e).as_str());
                        err.push_str("Please check your config file.");
                        Err(err)
                    }
                }
            }

            pub fn used_options(&self) -> PartialConfig {
                PartialConfig {
                    $(
                        $i: if self.$i.0.get() {
                                Some(self.$i.2.clone())
                            } else {
                                None
                            },
                    )+
                }
            }

            pub fn all_options(&self) -> PartialConfig {
                PartialConfig {
                    $(
                        $i: Some(self.$i.2.clone()),
                    )+
                }
            }

            pub fn override_value(&mut self, key: &str, val: &str)
            {
                match key {
                    $(
                        stringify!($i) => {
                            self.$i.2 = val.parse::<$ty>()
                                .expect(&format!("Failed to parse override for {} (\"{}\") as a {}",
                                                 stringify!($i),
                                                 val,
                                                 stringify!($ty)));
                        }
                    )+
                    _ => panic!("Unknown config key in override: {}", key)
                }

                if key == "use_small_heuristics" {
                    self.set_heuristics();
                }
            }

            /// Construct a `Config` from the toml file specified at `file_path`.
            ///
            /// This method only looks at the provided path, for a method that
            /// searches parents for a `rustfmt.toml` see `from_resolved_toml_path`.
            ///
            /// Return a `Config` if the config could be read and parsed from
            /// the file, Error otherwise.
            pub fn from_toml_path(file_path: &Path) -> Result<Config, Error> {
                let mut file = File::open(&file_path)?;
                let mut toml = String::new();
                file.read_to_string(&mut toml)?;
                Config::from_toml(&toml).map_err(|err| Error::new(ErrorKind::InvalidData, err))
            }

            /// Resolve the config for input in `dir`.
            ///
            /// Searches for `rustfmt.toml` beginning with `dir`, and
            /// recursively checking parents of `dir` if no config file is found.
            /// If no config file exists in `dir` or in any parent, a
            /// default `Config` will be returned (and the returned path will be empty).
            ///
            /// Returns the `Config` to use, and the path of the project file if there was
            /// one.
            pub fn from_resolved_toml_path(dir: &Path) -> Result<(Config, Option<PathBuf>), Error> {

                /// Try to find a project file in the given directory and its parents.
                /// Returns the path of a the nearest project file if one exists,
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
                            _ => ()
                        }

                        // If the current directory has no parent, we're done searching.
                        if !current.pop() {
                            return Ok(None);
                        }
                    }
                }

                match resolve_project_file(dir)? {
                    None => Ok((Config::default(), None)),
                    Some(path) => Config::from_toml_path(&path).map(|config| (config, Some(path))),
                }
            }


            pub fn print_docs() {
                use std::cmp;
                const HIDE_OPTIONS: [&str; 3] = ["verbose", "file_lines", "width_heuristics"];
                let max = 0;
                $( let max = cmp::max(max, stringify!($i).len()+1); )+
                let mut space_str = String::with_capacity(max);
                for _ in 0..max {
                    space_str.push(' ');
                }
                println!("Configuration Options:");
                $(
                    let name_raw = stringify!($i);

                    if !HIDE_OPTIONS.contains(&name_raw) {
                        let mut name_out = String::with_capacity(max);
                        for _ in name_raw.len()..max-1 {
                            name_out.push(' ')
                        }
                        name_out.push_str(name_raw);
                        name_out.push(' ');
                        println!("{}{} Default: {:?}",
                                name_out,
                                <$ty>::doc_hint(),
                                $def);
                        $(
                            println!("{}{}", space_str, $dstring);
                        )+
                        println!();
                    }
                )+
            }

            fn set_heuristics(&mut self) {
                if self.use_small_heuristics.2 {
                    self.set().width_heuristics(WidthHeuristics::default());
                } else {
                    self.set().width_heuristics(WidthHeuristics::null());
                }
            }
        }

        // Template for the default configuration
        impl Default for Config {
            fn default() -> Config {
                Config {
                    $(
                        $i: (Cell::new(false), false, $def, $stb),
                    )+
                }
            }
        }
    )
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

create_config! {
    // Fundamental stuff
    max_width: usize, 100, true, "Maximum width of each line";
    hard_tabs: bool, false, true, "Use tab characters for indentation, spaces for alignment";
    tab_spaces: usize, 4, true, "Number of spaces per tab";
    newline_style: NewlineStyle, NewlineStyle::Unix, true, "Unix or Windows line endings";
    indent_style: IndentStyle, IndentStyle::Block, false, "How do we indent expressions or items.";
    use_small_heuristics: bool, true, false, "Whether to use different formatting for items and\
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
    width_heuristics: WidthHeuristics, WidthHeuristics::default(), false,
        "'small' heuristic values";
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
