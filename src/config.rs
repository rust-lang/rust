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
pub use issues::ReportTactic;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum NewlineStyle {
    Windows, // \r\n
    Unix, // \n
}

impl_enum_decodable!(NewlineStyle, Windows, Unix);

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum BraceStyle {
    AlwaysNextLine,
    PreferSameLine,
    // Prefer same line except where there is a where clause, in which case force
    // the brace to the next line.
    SameLineWhere,
}

impl_enum_decodable!(BraceStyle, AlwaysNextLine, PreferSameLine, SameLineWhere);

// How to indent a function's return type.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ReturnIndent {
    // Aligned with the arguments
    WithArgs,
    // Aligned with the where clause
    WithWhereClause,
}

impl_enum_decodable!(ReturnIndent, WithArgs, WithWhereClause);

// How to stle a struct literal.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum StructLitStyle {
    // First line on the same line as the opening brace, all lines aligned with
    // the first line.
    Visual,
    // First line is on a new line and all lines align with block indent.
    Block,
    // FIXME Maybe we should also have an option to align types.
}

impl_enum_decodable!(StructLitStyle, Visual, Block);

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum BlockIndentStyle {
    // Same level as parent.
    Inherit,
    // One level deeper than parent.
    Tabbed,
    // Aligned with block open.
    Visual,
}

impl_enum_decodable!(BlockIndentStyle, Inherit, Tabbed, Visual);

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Density {
    // Fit as much on one line as possible.
    Compressed,
    // Use more lines.
    Tall,
}

impl_enum_decodable!(Density, Compressed, Tall);

impl Density {
    pub fn to_list_tactic(self) -> ListTactic {
        match self {
            Density::Compressed => ListTactic::Mixed,
            Density::Tall => ListTactic::HorizontalVertical,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum MultilineStyle {
    // Use horizontal layout if it fits in one line, fall back to vertical
    PreferSingle,
    // Use vertical layout
    ForceMulti,
}


impl_enum_decodable!(MultilineStyle, PreferSingle, ForceMulti);

impl MultilineStyle {
    pub fn to_list_tactic(self) -> ListTactic {
        match self {
            MultilineStyle::PreferSingle => ListTactic::HorizontalVertical,
            MultilineStyle::ForceMulti => ListTactic::Vertical,
        }
    }
}

macro_rules! create_config {
    ($($i:ident: $ty:ty, $dstring: tt),+ $(,)*) => (
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

        pub struct ConfigHelpItem {
            option_name: &'static str,
            doc_string : &'static str,
            variant_names: String,
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
        }

        impl Config {

            fn fill_from_parsed_config(mut self, parsed: &ParsedConfig) -> Config {
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
                Config::default().fill_from_parsed_config(&parsed_config)
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

            pub fn get_docs() -> Vec<ConfigHelpItem> {
                let mut options: Vec<ConfigHelpItem> = Vec::new();
                $(
                    options.push(ConfigHelpItem {
                        option_name: stringify!($i),
                        doc_string: stringify!($dstring),
                        variant_names: <$ty>::get_variant_names(),
                    });
                )+
                options
            }
        }
    )
}

create_config! {
    max_width: usize, "Maximum width of each line",
    ideal_width: usize, "Ideal width of each line",
    leeway: usize, "Leeway of line width",
    tab_spaces: usize, "Number of spaces per tab",
    newline_style: NewlineStyle, "Unix or Windows line endings",
    fn_brace_style: BraceStyle, "Brace style for functions",
    fn_return_indent: ReturnIndent, "Location of return type in function declaration",
    fn_args_paren_newline: bool, "If function argument parenthases goes on a newline",
    fn_args_density: Density, "Argument density in functions",
    fn_args_layout: StructLitStyle, "Layout of function arguments",
    fn_arg_indent: BlockIndentStyle, "Indent on function arguments",
    // Should we at least try to put the where clause on the same line as the rest of the
    // function decl?
    where_density: Density, "Density of a where clause",
    // Visual will be treated like Tabbed
    where_indent: BlockIndentStyle, "Indentation of a where clause",
    where_layout: ListTactic, "Element layout inside a where clause",
    where_pred_indent: BlockIndentStyle, "Indentation style of a where predicate",
    generics_indent: BlockIndentStyle, "Indentation of generics",
    struct_trailing_comma: SeparatorTactic, "If there is a trailing comma on structs",
    struct_lit_trailing_comma: SeparatorTactic, "If there is a trailing comma on literal structs",
    struct_lit_style: StructLitStyle, "Style of struct definition",
    struct_lit_multiline_style: MultilineStyle, "Multilline style on literal structs",
    enum_trailing_comma: bool, "Put a trailing comma on enum declarations",
    report_todo: ReportTactic, "Report all occurences of TODO in source file comments",
    report_fixme: ReportTactic, "Report all occurences of FIXME in source file comments",
    // Alphabetically, case sensitive.
    reorder_imports: bool, "Reorder import statements alphabetically",
    single_line_if_else: bool, "Put else on same line as closing brace for if statements",
    format_strings: bool, "Format string literals, or leave as is",
    chains_overflow_last: bool, "Allow last call in method chain to break the line",
    take_source_hints: bool, "Retain some formatting characteristics from the source code",
}

impl Default for Config {
    fn default() -> Config {
        Config {
            max_width: 100,
            ideal_width: 80,
            leeway: 5,
            tab_spaces: 4,
            newline_style: NewlineStyle::Unix,
            fn_brace_style: BraceStyle::SameLineWhere,
            fn_return_indent: ReturnIndent::WithArgs,
            fn_args_paren_newline: true,
            fn_args_density: Density::Tall,
            fn_args_layout: StructLitStyle::Visual,
            fn_arg_indent: BlockIndentStyle::Visual,
            where_density: Density::Tall,
            where_indent: BlockIndentStyle::Tabbed,
            where_layout: ListTactic::Vertical,
            where_pred_indent: BlockIndentStyle::Visual,
            generics_indent: BlockIndentStyle::Visual,
            struct_trailing_comma: SeparatorTactic::Vertical,
            struct_lit_trailing_comma: SeparatorTactic::Vertical,
            struct_lit_style: StructLitStyle::Block,
            struct_lit_multiline_style: MultilineStyle::PreferSingle,
            enum_trailing_comma: true,
            report_todo: ReportTactic::Always,
            report_fixme: ReportTactic::Never,
            reorder_imports: false,
            single_line_if_else: false,
            format_strings: true,
            chains_overflow_last: true,
            take_source_hints: true,
        }
    }
}
