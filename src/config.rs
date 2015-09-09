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

use {NewlineStyle, BraceStyle, ReturnIndent, StructLitStyle};
use lists::{SeparatorTactic, ListTactic};
use issues::ReportTactic;

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
    ($($i:ident: $ty:ty),+ $(,)*) => (
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
        }
    )
}

create_config! {
    max_width: usize,
    ideal_width: usize,
    leeway: usize,
    tab_spaces: usize,
    newline_style: NewlineStyle,
    fn_brace_style: BraceStyle,
    fn_return_indent: ReturnIndent,
    fn_args_paren_newline: bool,
    fn_args_density: Density,
    fn_args_layout: StructLitStyle,
    fn_arg_indent: BlockIndentStyle,
    where_density: Density, // Should we at least try to put the where clause on
                            // the same line as the rest of the function decl?
    where_indent: BlockIndentStyle, // Visual will be treated like Tabbed
    where_layout: ListTactic,
    where_pred_indent: BlockIndentStyle,
    generics_indent: BlockIndentStyle,
    struct_trailing_comma: SeparatorTactic,
    struct_lit_trailing_comma: SeparatorTactic,
    struct_lit_style: StructLitStyle,
    struct_lit_multiline_style: MultilineStyle,
    enum_trailing_comma: bool,
    report_todo: ReportTactic,
    report_fixme: ReportTactic,
    reorder_imports: bool, // Alphabetically, case sensitive.
    single_line_if_else: bool,
    format_strings: bool,
    chains_overflow_last: bool,
    take_source_hints: bool, // Retain some formatting characteristics from
                             // the source code.
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
