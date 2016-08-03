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

use file_lines::FileLines;
use lists::{SeparatorTactic, ListTactic};
use std::io::Write;

macro_rules! configuration_option_enum{
    ($e:ident: $( $x:ident ),+ $(,)*) => {
        #[derive(Copy, Clone, Eq, PartialEq, Debug)]
        pub enum $e {
            $( $x ),+
        }

        impl_enum_decodable!($e, $( $x ),+);
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
    // K&R/Stroustrup style, Rust community default
    AlwaysSameLine,
    // Allman style
    AlwaysNextLine,
}

configuration_option_enum! { ElseIfBraceStyle:
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

// How to style a struct literal.
configuration_option_enum! { StructLitStyle:
    // First line on the same line as the opening brace, all lines aligned with
    // the first line.
    Visual,
    // First line is on a new line and all lines align with block indent.
    Block,
    // FIXME Maybe we should also have an option to align types.
}

// How to style fn args.
configuration_option_enum! { FnArgLayoutStyle:
    // First line on the same line as the opening brace, all lines aligned with
    // the first line.
    Visual,
    // Put args on one line if they fit, or start a new line with block indent.
    Block,
    // First line is on a new line and all lines align with block indent.
    BlockAlways,
}

configuration_option_enum! { BlockIndentStyle:
    // Same level as parent.
    Inherit,
    // One level deeper than parent.
    Tabbed,
    // Aligned with block open.
    Visual,
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
    // Backsup the original file and overwrites the orignal.
    Replace,
    // Overwrites original file without backup.
    Overwrite,
    // Write the output to stdout.
    Display,
    // Write the diff to stdout.
    Diff,
    // Display how much of the input file was processed
    Coverage,
    // Unfancy stdout
    Plain,
    // Output a checkstyle XML file.
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
        #[derive(RustcDecodable, Clone)]
        pub struct Config {
            $(pub $i: $ty),+
        }

        // Just like the Config struct but with each property wrapped
        // as Option<T>. This is used to parse a rustfmt.toml that doesn't
        // specity all properties of `Config`.
        // We first parse into `ParsedConfig`, then create a default `Config`
        // and overwrite the properties with corresponding values from `ParsedConfig`
        #[derive(RustcDecodable, Clone)]
        pub struct ParsedConfig {
            $(pub $i: Option<$ty>),+
        }

        impl Config {

            fn fill_from_parsed_config(mut self, parsed: ParsedConfig) -> Config {
            $(
                if let Some(val) = parsed.$i {
                    self.$i = val;
                }
            )+
                self
            }

            pub fn from_toml(toml: &str) -> Config {
                let parsed = toml.parse().expect("Could not parse TOML");
                let parsed_config:ParsedConfig = match toml::decode(parsed) {
                    Some(decoded) => decoded,
                    None => {
                        msg!("Decoding config file failed. Config:\n{}", toml);
                        let parsed: toml::Value = toml.parse().expect("Could not parse TOML");
                        msg!("\n\nParsed:\n{:?}", parsed);
                        panic!();
                    }
                };
                Config::default().fill_from_parsed_config(parsed_config)
            }

            pub fn override_value(&mut self, key: &str, val: &str) {
                match key {
                    $(
                        stringify!($i) => {
                            self.$i = val.parse::<$ty>()
                                .expect(&format!("Failed to parse override for {} (\"{}\") as a {}",
                                                 stringify!($i),
                                                 val,
                                                 stringify!($ty)));
                        }
                    )+
                    _ => panic!("Unknown config key in override: {}", key)
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
                        $i: $def,
                    )+
                }
            }
        }
    )
}

create_config! {
    verbose: bool, false, "Use verbose output";
    skip_children: bool, false, "Don't reformat out of line modules";
    file_lines: FileLines, FileLines::all(),
        "Lines to format; this is not supported in rustfmt.toml, and can only be specified \
         via the --file-lines option";
    max_width: usize, 100, "Maximum width of each line";
    ideal_width: usize, 80, "Ideal width of each line";
    tab_spaces: usize, 4, "Number of spaces per tab";
    fn_call_width: usize, 60,
        "Maximum width of the args of a function call before falling back to vertical formatting";
    struct_lit_width: usize, 16,
        "Maximum width in the body of a struct lit before falling back to vertical formatting";
    struct_variant_width: usize, 35,
        "Maximum width in the body of a struct variant before falling back to vertical formatting";
    force_explicit_abi: bool, true, "Always print the abi for extern items";
    newline_style: NewlineStyle, NewlineStyle::Unix, "Unix or Windows line endings";
    fn_brace_style: BraceStyle, BraceStyle::SameLineWhere, "Brace style for functions";
    item_brace_style: BraceStyle, BraceStyle::SameLineWhere, "Brace style for structs and enums";
    else_if_brace_style: ElseIfBraceStyle, ElseIfBraceStyle::AlwaysSameLine,
        "Brace style for if, else if, and else constructs";
    control_brace_style: ControlBraceStyle, ControlBraceStyle::AlwaysSameLine,
        "Brace style for match, loop, for, and while constructs";
    impl_empty_single_line: bool, true, "Put empty-body implementations on a single line";
    fn_empty_single_line: bool, true, "Put empty-body functions on a single line";
    fn_single_line: bool, false, "Put single-expression functions on a single line";
    fn_return_indent: ReturnIndent, ReturnIndent::WithArgs,
        "Location of return type in function declaration";
    fn_args_paren_newline: bool, true, "If function argument parenthesis goes on a newline";
    fn_args_density: Density, Density::Tall, "Argument density in functions";
    fn_args_layout: FnArgLayoutStyle, FnArgLayoutStyle::Visual, "Layout of function arguments";
    fn_arg_indent: BlockIndentStyle, BlockIndentStyle::Visual, "Indent on function arguments";
    type_punctuation_density: TypeDensity, TypeDensity::Wide,
        "Determines if '+' or '=' are wrapped in spaces in the punctuation of types";
    // Should we at least try to put the where clause on the same line as the rest of the
    // function decl?
    where_density: Density, Density::CompressedIfEmpty, "Density of a where clause";
    // Visual will be treated like Tabbed
    where_indent: BlockIndentStyle, BlockIndentStyle::Tabbed, "Indentation of a where clause";
    where_layout: ListTactic, ListTactic::Vertical, "Element layout inside a where clause";
    where_pred_indent: BlockIndentStyle, BlockIndentStyle::Visual,
        "Indentation style of a where predicate";
    where_trailing_comma: bool, false, "Put a trailing comma on where clauses";
    generics_indent: BlockIndentStyle, BlockIndentStyle::Visual, "Indentation of generics";
    struct_trailing_comma: SeparatorTactic, SeparatorTactic::Vertical,
        "If there is a trailing comma on structs";
    struct_lit_trailing_comma: SeparatorTactic, SeparatorTactic::Vertical,
        "If there is a trailing comma on literal structs";
    struct_lit_style: StructLitStyle, StructLitStyle::Block, "Style of struct definition";
    struct_lit_multiline_style: MultilineStyle, MultilineStyle::PreferSingle,
        "Multiline style on literal structs";
    enum_trailing_comma: bool, true, "Put a trailing comma on enum declarations";
    report_todo: ReportTactic, ReportTactic::Never,
        "Report all, none or unnumbered occurrences of TODO in source file comments";
    report_fixme: ReportTactic, ReportTactic::Never,
        "Report all, none or unnumbered occurrences of FIXME in source file comments";
    chain_base_indent: BlockIndentStyle, BlockIndentStyle::Tabbed, "Indent on chain base";
    chain_indent: BlockIndentStyle, BlockIndentStyle::Tabbed, "Indentation of chain";
    chains_overflow_last: bool, true, "Allow last call in method chain to break the line";
    reorder_imports: bool, false, "Reorder import statements alphabetically";
    reorder_imported_names: bool, false,
        "Reorder lists of names in import statements alphabetically";
    single_line_if_else_max_width: usize, 50, "Maximum line length for single line if-else \
                                                expressions. A value of zero means always break \
                                                if-else expressions.";
    format_strings: bool, true, "Format string literals where necessary";
    force_format_strings: bool, false, "Always format string literals";
    take_source_hints: bool, true, "Retain some formatting characteristics from the source code";
    hard_tabs: bool, false, "Use tab characters for indentation, spaces for alignment";
    wrap_comments: bool, false, "Break comments to fit on the line";
    normalize_comments: bool, true, "Convert /* */ comments to // comments where possible";
    wrap_match_arms: bool, true, "Wrap multiline match arms in blocks";
    match_block_trailing_comma: bool, false,
        "Put a trailing comma after a block based match arm (non-block arms are not affected)";
    match_wildcard_trailing_comma: bool, true, "Put a trailing comma after a wildcard arm";
    closure_block_indent_threshold: isize, 5, "How many lines a closure must have before it is \
                                               block indented. -1 means never use block indent.";
    space_before_type_annotation: bool, false,
        "Leave a space before the colon in a type annotation";
    space_before_bound: bool, false, "Leave a space before the colon in a trait or lifetime bound";
    use_try_shorthand: bool, false, "Replace uses of the try! macro by the ? shorthand";
    write_mode: WriteMode, WriteMode::Replace,
        "What Write Mode to use when none is supplied: Replace, Overwrite, Display, Diff, Coverage";
}
