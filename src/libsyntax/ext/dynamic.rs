// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
Dynamically loading syntax extensions.

Implementation of the #[syntax_extension="filename"] attribute, that
allows an external Rust library to provide custom syntax
extensions. This library should define public function called
`register_syntax_extensions` (with #[no_mangle], to preserve the
name), with type `fn (@ExtCtxt) -> ~[(@~str, @Transformer)]` (both
`ExtCtxt` and `Transformer` are defined in `ext::base`). This function
returns a vector of the new syntax extensions and their names.

The `#[syntax_extension]` attribute current only works at the crate
level, is highly experimental and requires the `-Z
dynamic-syntax-extensions` flag.

# Example

## `my_ext.rs`
~~~
extern mod syntax;
use syntax::ast;
use syntax::parse;
use syntax::codemap::span;
use syntax::ext::base::{ExtCtxt, builtin_normal_tt, MacResult, MRExpr, Transformer};
use syntax::ext::build::AstBuilder;

#[no_mangle]
pub fn register_syntax_extensions(_cx: @ExtCtxt) -> ~[(@~str, @Transformer)] {
    ~[(@~"my_macro", builtin_normal_tt(my_macro))]
}

fn my_macro(cx: @ExtCtxt, sp: span, _tts: &[ast::token_tree]) -> MacResult {
    MRExpr(cx.expr_str(sp, ~"hello world"))
}
~~~

## `main.rs`
Compiled with `-Z dynamic-syntax-extensions`
~~~
#[syntax_extension="libmy_ext-<hash>-<version>.so"]

fn main() {
    println(my_macro!())
}
~~~

The exact file name of the library is required, and it needs to either
be in a directory that the library loader will search in (e.g. in
`LD_LIBRARY_PATH`), or be specified as a file path.

*/

use core::prelude::*;
use core::unstable::dynamic_lib::DynamicLib;
use ast;
use attr;
use codemap::span;
use ext::base::{ExtCtxt, SyntaxEnv, Transformer};

static REGISTER_FN: &'static str = "register_syntax_extensions";
type RegisterFnType = extern fn(@ExtCtxt) -> ~[(@~str, @Transformer)];

// the return value needs to be stored until all expansion has
// happened, otherwise syntax extensions will be calling into a
// library that has been closed, causing segfaults
pub fn load_dynamic_crate(cx: @ExtCtxt, table: @mut SyntaxEnv, c: @ast::crate, enabled: bool)
    -> ~[DynamicLib] {
    do vec::build |push| {
        for attr::find_attrs_by_name(c.node.attrs, "syntax_extension").each |attr| {
            if !enabled {
                cx.span_fatal(attr.span,
                              "#[syntax_extension] is experimental and disabled by \
                               default; use `-Z dynamic-syntax-extensions` to enable");
            }

            match attr::get_meta_item_value_str(attr::attr_meta(*attr)) {
                Some(value) => {
                    match load_dynamic(cx, *value, table) {
                        Ok(lib) => push(lib),
                        Err(err) => {
                            cx.span_err(attr.span,
                                        fmt!("could not load syntax extensions from `%s`: %s",
                                             *value, err));
                        }
                    }
                }
                None => { cx.span_err(attr.span, "Expecting `syntax_extension=\"filename\"`") }
            }
        }
    }
}

fn load_dynamic(cx: @ExtCtxt, name: &str, table: @mut SyntaxEnv) -> Result<DynamicLib, ~str> {
    match DynamicLib::open(name) {
        Ok(lib) => {
            let mut error = None;
            { // borrow checker complains about lib & sym without this block
                let sym = lib.get_symbol::<RegisterFnType>(REGISTER_FN);
                match sym {
                    Ok(func) => {
                        // should need unsafe { } (#3080)
                        let syn_exts = (*func.get())(cx);
                        for syn_exts.each |&(name, transformer)| {
                            table.insert(name, transformer);
                        }
                    }
                    Err(err) => { error = Some(err); }
                }
            }
            match error {
                None => Ok(lib),
                Some(err) => Err(err)
            }
        }
        Err(err) => Err(err)
    }
}