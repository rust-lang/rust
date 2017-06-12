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

use std::cell::Cell;
use std::fs;
use std::fs::File;
use std::env;
use std::io::{Error, ErrorKind, Read};
use std::path::{Path, PathBuf};

use file_lines::FileLines;
use lists::{SeparatorTactic, ListTactic};

macro_rules! configuration_option_enum{
    ($e:ident: $( $x:ident ),+ $(,)*) => {
        #[derive(Copy, Clone, Eq, PartialEq, Debug)]
        pub enum $e {
            $( $x ),+
        }

        impl_enum_serialize_and_deserialize!($e, $( $x ),+);
    }
}

configuration_option_enum! { Style:
    Rfc, // Follow the style RFCs style.
    Legacy, // Follow the traditional Rustfmt style.
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

// How to indent a function's return type.
configuration_option_enum! { ReturnIndent:
    // Aligned with the arguments
    WithArgs,
    // Aligned with the where clause
    WithWhereClause,
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
    // Try to compress if the body is empty.
    CompressedIfEmpty,
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
            Density::Tall |
            Density::CompressedIfEmpty => ListTactic::HorizontalVertical,
            Density::Vertical => ListTactic::Vertical,
        }
    }
}

configuration_option_enum! { LicensePolicy:
    // Do not place license text at top of files
    NoLicense,
    // Use the text in "license" field as the license
    TextLicense,
    // Use a text file as the license text
    FileLicense,
}

configuration_option_enum! { MultilineStyle:
    // Use horizontal layout if it fits in one line, fall back to vertical
    PreferSingle,
    // Use vertical layout
    ForceMulti,
}

impl MultilineStyle {
    pub fn to_list_tactic(self) -> ListTactic {
        match self {
            MultilineStyle::PreferSingle => ListTactic::HorizontalVertical,
            MultilineStyle::ForceMulti => ListTactic::Vertical,
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
    ($($i:ident: $ty:ty, $def:expr, $( $dstring:expr ),+ );+ $(;)*) => (
        #[derive(Clone)]
        pub struct Config {
            // For each config item, we store a bool indicating whether it has
            // been accessed and the value.
            $($i: (Cell<bool>, $ty)),+
        }

        // Just like the Config struct but with each property wrapped
        // as Option<T>. This is used to parse a rustfmt.toml that doesn't
        // specity all properties of `Config`.
        // We first parse into `PartialConfig`, then create a default `Config`
        // and overwrite the properties with corresponding values from `PartialConfig`.
        #[derive(Deserialize, Serialize, Clone)]
        pub struct PartialConfig {
            $(pub $i: Option<$ty>),+
        }

        impl PartialConfig {
            pub fn to_toml(&self) -> Result<String, String> {
                // file_lines can't be specified in TOML
                let mut cloned = self.clone();
                cloned.file_lines = None;

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
                (self.0).$i.1 = value;
            }
            )+
        }

        impl Config {

            $(
            pub fn $i(&self) -> $ty {
                self.$i.0.set(true);
                self.$i.1.clone()
            }
            )+

            pub fn set<'a>(&'a mut self) -> ConfigSetter<'a> {
                ConfigSetter(self)
            }

            fn fill_from_parsed_config(mut self, parsed: PartialConfig) -> Config {
            $(
                if let Some(val) = parsed.$i {
                    self.$i.1 = val;
                }
            )+
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
                    for (key, _) in table {
                        match &**key {
                            $(
                                stringify!($i) => (),
                            )+
                                _ => {
                                    let msg =
                                        &format!("Warning: Unknown configuration option `{}`\n",
                                                 key);
                                    err.push_str(msg)
                                }
                        }
                    }
                }
                match parsed.try_into() {
                    Ok(parsed_config) =>
                        Ok(Config::default().fill_from_parsed_config(parsed_config)),
                    Err(e) => {
                        err.push_str("Error: Decoding config file failed:\n");
                        err.push_str(format!("{}\n", e).as_str());
                        err.push_str("Please check your config file.\n");
                        Err(err)
                    }
                }
            }

            pub fn used_options(&self) -> PartialConfig {
                PartialConfig {
                    $(
                        $i: if self.$i.0.get() {
                                Some(self.$i.1.clone())
                            } else {
                                None
                            },
                    )+
                }
            }

            pub fn all_options(&self) -> PartialConfig {
                PartialConfig {
                    $(
                        $i: Some(self.$i.1.clone()),
                    )+
                }
            }

            pub fn override_value(&mut self, key: &str, val: &str)
            {
                match key {
                    $(
                        stringify!($i) => {
                            self.$i.1 = val.parse::<$ty>()
                                .expect(&format!("Failed to parse override for {} (\"{}\") as a {}",
                                                 stringify!($i),
                                                 val,
                                                 stringify!($ty)));
                        }
                    )+
                    _ => panic!("Unknown config key in override: {}", key)
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
                let max = 0;
                $( let max = cmp::max(max, stringify!($i).len()+1); )+
                let mut space_str = String::with_capacity(max);
                for _ in 0..max {
                    space_str.push(' ');
                }
                println!("Configuration Options:");
                $(
                    let name_raw = stringify!($i);
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
                    println!("");
                )+
            }
        }

        // Template for the default configuration
        impl Default for Config {
            fn default() -> Config {
                Config {
                    $(
                        $i: (Cell::new(false), $def),
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
    const CONFIG_FILE_NAMES: [&'static str; 2] = [".rustfmt.toml", "rustfmt.toml"];
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
    verbose: bool, false, "Use verbose output";
    disable_all_formatting: bool, false, "Don't reformat anything";
    skip_children: bool, false, "Don't reformat out of line modules";
    file_lines: FileLines, FileLines::all(),
        "Lines to format; this is not supported in rustfmt.toml, and can only be specified \
         via the --file-lines option";
    max_width: usize, 100, "Maximum width of each line";
    error_on_line_overflow: bool, true, "Error if unable to get all lines within max_width";
    tab_spaces: usize, 4, "Number of spaces per tab";
    fn_call_width: usize, 60,
        "Maximum width of the args of a function call before falling back to vertical formatting";
    struct_lit_width: usize, 18,
        "Maximum width in the body of a struct lit before falling back to vertical formatting";
    struct_variant_width: usize, 35,
        "Maximum width in the body of a struct variant before falling back to vertical formatting";
    force_explicit_abi: bool, true, "Always print the abi for extern items";
    newline_style: NewlineStyle, NewlineStyle::Unix, "Unix or Windows line endings";
    fn_brace_style: BraceStyle, BraceStyle::SameLineWhere, "Brace style for functions";
    item_brace_style: BraceStyle, BraceStyle::SameLineWhere, "Brace style for structs and enums";
    control_style: Style, Style::Rfc, "Indent style for control flow statements";
    control_brace_style: ControlBraceStyle, ControlBraceStyle::AlwaysSameLine,
        "Brace style for control flow constructs";
    impl_empty_single_line: bool, true, "Put empty-body implementations on a single line";
    trailing_comma: SeparatorTactic, SeparatorTactic::Vertical,
        "How to handle trailing commas for lists";
    fn_empty_single_line: bool, true, "Put empty-body functions on a single line";
    fn_single_line: bool, false, "Put single-expression functions on a single line";
    fn_return_indent: ReturnIndent, ReturnIndent::WithArgs,
        "Location of return type in function declaration";
    fn_args_paren_newline: bool, false, "If function argument parenthesis goes on a newline";
    fn_args_density: Density, Density::Tall, "Argument density in functions";
    fn_args_layout: IndentStyle, IndentStyle::Block,
        "Layout of function arguments and tuple structs";
    array_layout: IndentStyle, IndentStyle::Block, "Indent on arrays";
    array_width: usize, 60,
        "Maximum width of an array literal before falling back to vertical formatting";
    type_punctuation_density: TypeDensity, TypeDensity::Wide,
        "Determines if '+' or '=' are wrapped in spaces in the punctuation of types";
    where_style: Style, Style::Rfc, "Overall strategy for where clauses";
    // TODO:
    // 1. Should we at least try to put the where clause on the same line as the rest of the
    // function decl?
    // 2. Currently options `Tall` and `Vertical` produce the same output.
    where_density: Density, Density::CompressedIfEmpty, "Density of a where clause";
    where_layout: ListTactic, ListTactic::Vertical, "Element layout inside a where clause";
    where_pred_indent: IndentStyle, IndentStyle::Visual,
        "Indentation style of a where predicate";
    generics_indent: IndentStyle, IndentStyle::Block, "Indentation of generics";
    struct_lit_style: IndentStyle, IndentStyle::Block, "Style of struct definition";
    struct_lit_multiline_style: MultilineStyle, MultilineStyle::PreferSingle,
        "Multiline style on literal structs";
    fn_call_style: IndentStyle, IndentStyle::Block, "Indentation for function calls, etc.";
    report_todo: ReportTactic, ReportTactic::Never,
        "Report all, none or unnumbered occurrences of TODO in source file comments";
    report_fixme: ReportTactic, ReportTactic::Never,
        "Report all, none or unnumbered occurrences of FIXME in source file comments";
    chain_indent: IndentStyle, IndentStyle::Block, "Indentation of chain";
    chain_one_line_max: usize, 60, "Maximum length of a chain to fit on a single line";
    chain_split_single_child: bool, false, "Split a chain with a single child if its length \
                                            exceeds `chain_one_line_max`";
    reorder_imports: bool, false, "Reorder import statements alphabetically";
    reorder_imports_in_group: bool, false, "Reorder import statements in group";
    reorder_imported_names: bool, false,
        "Reorder lists of names in import statements alphabetically";
    single_line_if_else_max_width: usize, 50, "Maximum line length for single line if-else \
                                                expressions. A value of zero means always break \
                                                if-else expressions.";
    format_strings: bool, false, "Format string literals where necessary";
    force_format_strings: bool, false, "Always format string literals";
    take_source_hints: bool, false, "Retain some formatting characteristics from the source code";
    hard_tabs: bool, false, "Use tab characters for indentation, spaces for alignment";
    wrap_comments: bool, false, "Break comments to fit on the line";
    comment_width: usize, 80, "Maximum length of comments. No effect unless wrap_comments = true";
    normalize_comments: bool, false, "Convert /* */ comments to // comments where possible";
    wrap_match_arms: bool, true, "Wrap multiline match arms in blocks";
    match_block_trailing_comma: bool, false,
        "Put a trailing comma after a block based match arm (non-block arms are not affected)";
    indent_match_arms: bool, true, "Indent match arms instead of keeping them at the same \
                                    indentation level as the match keyword";
    closure_block_indent_threshold: isize, 7, "How many lines a closure must have before it is \
                                               block indented. -1 means never use block indent.";
    space_before_type_annotation: bool, false,
        "Leave a space before the colon in a type annotation";
    space_after_type_annotation_colon: bool, true,
        "Leave a space after the colon in a type annotation";
    space_before_struct_lit_field_colon: bool, false,
        "Leave a space before the colon in a struct literal field";
    space_after_struct_lit_field_colon: bool, true,
        "Leave a space after the colon in a struct literal field";
    space_before_bound: bool, false, "Leave a space before the colon in a trait or lifetime bound";
    space_after_bound_colon: bool, true,
        "Leave a space after the colon in a trait or lifetime bound";
    spaces_around_ranges: bool, false, "Put spaces around the  .. and ... range operators";
    spaces_within_angle_brackets: bool, false, "Put spaces within non-empty generic arguments";
    spaces_within_square_brackets: bool, false, "Put spaces within non-empty square brackets";
    spaces_within_parens: bool, false, "Put spaces within non-empty parentheses";
    use_try_shorthand: bool, false, "Replace uses of the try! macro by the ? shorthand";
    write_mode: WriteMode, WriteMode::Replace,
        "What Write Mode to use when none is supplied: Replace, Overwrite, Display, Diff, Coverage";
    condense_wildcard_suffixes: bool, false, "Replace strings of _ wildcards by a single .. in \
                                              tuple patterns";
    combine_control_expr: bool, true, "Combine control expressions with funciton calls."
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

        let verbose = config.verbose();
        let skip_children = config.skip_children();

        let used_options = config.used_options();
        let toml = used_options.to_toml().unwrap();
        assert_eq!(
            toml,
            format!("verbose = {}\nskip_children = {}\n", verbose, skip_children)
        );
    }
}
