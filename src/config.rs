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
use lists::SeparatorTactic;
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

macro_rules! create_config {
    ($($i:ident: $ty:ty),+ $(,)*) => (
        #[derive(RustcDecodable, Clone)]
        pub struct Config {
            $(pub $i: $ty),+
        }

        impl Config {
            pub fn from_toml(toml: &str) -> Config {
                let parsed = toml.parse().unwrap();
                match toml::decode(parsed) {
                    Some(decoded) => decoded,
                    None => {
                        println!("Decoding config file failed. Config:\n{}", toml);
                        let parsed: toml::Value = toml.parse().unwrap();
                        println!("\n\nParsed:\n{:?}", parsed);
                        panic!();
                    }
                }
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
    struct_trailing_comma: SeparatorTactic,
    struct_lit_trailing_comma: SeparatorTactic,
    struct_lit_style: StructLitStyle,
    enum_trailing_comma: bool,
    report_todo: ReportTactic,
    report_fixme: ReportTactic,
    reorder_imports: bool, // Alphabetically, case sensitive.
    expr_indent_style: BlockIndentStyle,
    closure_indent_style: BlockIndentStyle,
    single_line_if_else: bool,
    format_strings: bool,
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
            struct_trailing_comma: SeparatorTactic::Vertical,
            struct_lit_trailing_comma: SeparatorTactic::Vertical,
            struct_lit_style: StructLitStyle::BlockIndent,
            enum_trailing_comma: true,
            report_todo: ReportTactic::Always,
            report_fixme: ReportTactic::Never,
            reorder_imports: false,
            expr_indent_style: BlockIndentStyle::Tabbed,
            closure_indent_style: BlockIndentStyle::Visual,
            single_line_if_else: false,
            format_strings: true,
        }
    }

}
