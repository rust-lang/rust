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

use {NewlineStyle, BraceStyle, ReturnIndent};
use lists::SeparatorTactic;
use issues::ReportTactic;

#[derive(RustcDecodable, Clone)]
pub struct Config {
    pub max_width: usize,
    pub ideal_width: usize,
    pub leeway: usize,
    pub tab_spaces: usize,
    pub newline_style: NewlineStyle,
    pub fn_brace_style: BraceStyle,
    pub fn_return_indent: ReturnIndent,
    pub fn_args_paren_newline: bool,
    pub struct_trailing_comma: bool,
    pub struct_lit_trailing_comma: SeparatorTactic,
    pub enum_trailing_comma: bool,
    pub report_todo: ReportTactic,
    pub report_fixme: ReportTactic,
}

impl Config {
    pub fn from_toml(toml: &str) -> Config {
        let parsed = toml.parse().unwrap();
        toml::decode(parsed).unwrap()
    }
}
