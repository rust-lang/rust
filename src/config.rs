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

#[derive(RustcDecodable)]
pub struct Config {
    pub max_width: usize,
    pub ideal_width: usize,
    pub leeway: usize,
    pub tab_spaces: usize,
    pub newline_style: ::NewlineStyle,
    pub fn_brace_style: ::BraceStyle,
    pub fn_return_indent: ::ReturnIndent,
}

impl Config {
    fn from_toml(toml: &str) -> Config {
        println!("About to parse: {}", toml);
        let parsed = toml.parse().unwrap();
        toml::decode(parsed).unwrap()
    }
}

pub fn set_config(toml: &str) {
    unsafe {
        ::CONFIG = Some(Config::from_toml(toml));
    }
}

macro_rules! config {
    ($name: ident) => {
        unsafe { ::CONFIG.as_ref().unwrap().$name }
    };
}
