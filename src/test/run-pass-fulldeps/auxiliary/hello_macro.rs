// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(plugin)]
#![feature(plugin_registrar)]
#![feature(rustc_private)]
#![plugin(proc_macro_plugin)]

extern crate rustc_plugin;
extern crate proc_macro_tokens;
extern crate syntax;

use syntax::ext::proc_macro_shim::prelude::*;
use proc_macro_tokens::prelude::*;

use rustc_plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("hello", hello);
}

// This macro is not very interesting, but it does contain delimited tokens with
// no content - `()` and `{}` - which has caused problems in the past.
fn hello<'cx>(cx: &'cx mut ExtCtxt, sp: Span, tts: &[TokenTree]) -> Box<MacResult + 'cx> {
    let output = qquote!({ fn hello() {} hello(); });
    build_block_emitter(cx, sp, output)
}
