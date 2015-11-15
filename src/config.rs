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

use lists::{SeparatorTactic, ListTactic};

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

// How to indent a function's return type.
configuration_option_enum! { ReturnIndent:
    // Aligned with the arguments
    WithArgs,
    // Aligned with the where clause
    WithWhereClause,
}

// How to stle a struct literal.
configuration_option_enum! { StructLitStyle:
    // First line on the same line as the opening brace, all lines aligned with
    // the first line.
    Visual,
    // First line is on a new line and all lines align with block indent.
    Block,
    // FIXME Maybe we should also have an option to align types.
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
}

impl Density {
    pub fn to_list_tactic(self) -> ListTactic {
        match self {
            Density::Compressed => ListTactic::Mixed,
            Density::Tall | Density::CompressedIfEmpty => ListTactic::HorizontalVertical,
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

// This trait and the following impl blocks are there so that we an use
// UCFS inside the get_docs() function on types for configs.
pub trait ConfigType {
    fn get_variant_names() -> String;
}

impl ConfigType for bool {
    fn get_variant_names() -> String {
        String::from("<boolean>")
    }
}

impl ConfigType for usize {
    fn get_variant_names() -> String {
        String::from("<unsigned integer>")
    }
}

impl ConfigType for String {
    fn get_variant_names() -> String {
        String::from("<string>")
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
                let parsed = toml.parse().unwrap();
                let parsed_config:ParsedConfig = match toml::decode(parsed) {
                    Some(decoded) => decoded,
                    None => {
                        println!("Decoding config file failed. Config:\n{}", toml);
                        let parsed: toml::Value = toml.parse().unwrap();
                        println!("\n\nParsed:\n{:?}", parsed);
                        panic!();
                    }
                };
                Config::default().fill_from_parsed_config(parsed_config)
            }

            pub fn override_value(&mut self, key: &str, val: &str) {
                match key {
                    $(
                        stringify!($i) => {
                            self.$i = val.parse::<$ty>().unwrap();
                        }
                    )+
                    _ => panic!("Bad config key!")
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
                             <$ty>::get_variant_names(),
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
    max_width: usize, 100, "Maximum width of each line";
    ideal_width: usize, 80, "Ideal width of each line";
    tab_spaces: usize, 4, "Number of spaces per tab";
    fn_call_width: usize, 60,
        "Maximum width of the args of a function call before falling back to vertical formatting";
    struct_lit_width: usize, 16,
        "Maximum width in the body of a struct lit before falling back to vertical formatting";
    newline_style: NewlineStyle, NewlineStyle::Unix, "Unix or Windows line endings";
    fn_brace_style: BraceStyle, BraceStyle::SameLineWhere, "Brace style for functions";
    fn_return_indent: ReturnIndent, ReturnIndent::WithArgs,
        "Location of return type in function declaration";
    fn_args_paren_newline: bool, true, "If function argument parenthesis goes on a newline";
    fn_args_density: Density, Density::Tall, "Argument density in functions";
    fn_args_layout: StructLitStyle, StructLitStyle::Visual, "Layout of function arguments";
    fn_arg_indent: BlockIndentStyle, BlockIndentStyle::Visual, "Indent on function arguments";
    // Should we at least try to put the where clause on the same line as the rest of the
    // function decl?
    where_density: Density, Density::CompressedIfEmpty, "Density of a where clause";
    // Visual will be treated like Tabbed
    where_indent: BlockIndentStyle, BlockIndentStyle::Tabbed, "Indentation of a where clause";
    where_layout: ListTactic, ListTactic::Vertical, "Element layout inside a where clause";
    where_pred_indent: BlockIndentStyle, BlockIndentStyle::Visual,
        "Indentation style of a where predicate";
    generics_indent: BlockIndentStyle, BlockIndentStyle::Visual, "Indentation of generics";
    struct_trailing_comma: SeparatorTactic, SeparatorTactic::Vertical,
        "If there is a trailing comma on structs";
    struct_lit_trailing_comma: SeparatorTactic, SeparatorTactic::Vertical,
        "If there is a trailing comma on literal structs";
    struct_lit_style: StructLitStyle, StructLitStyle::Block, "Style of struct definition";
    struct_lit_multiline_style: MultilineStyle, MultilineStyle::PreferSingle,
        "Multiline style on literal structs";
    enum_trailing_comma: bool, true, "Put a trailing comma on enum declarations";
    report_todo: ReportTactic, ReportTactic::Always,
        "Report all, none or unnumbered occurrences of TODO in source file comments";
    report_fixme: ReportTactic, ReportTactic::Never,
        "Report all, none or unnumbered occurrences of FIXME in source file comments";
    chain_base_indent: BlockIndentStyle, BlockIndentStyle::Visual, "Indent on chain base";
    chain_indent: BlockIndentStyle, BlockIndentStyle::Visual, "Indentation of chain";
    reorder_imports: bool, false, "Reorder import statements alphabetically";
    single_line_if_else: bool, false, "Put else on same line as closing brace for if statements";
    format_strings: bool, true, "Format string literals, or leave as is";
    chains_overflow_last: bool, true, "Allow last call in method chain to break the line";
    take_source_hints: bool, true, "Retain some formatting characteristics from the source code";
    hard_tabs: bool, false, "Use tab characters for indentation, spaces for alignment";
    wrap_comments: bool, false, "Break comments to fit on the line";
}
