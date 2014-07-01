// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;
use std::collections::HashMap;
use std::gc::Gc;
use ast;
use ast::{Ident, Name, TokenTree};
use codemap::Span;
use ext::base::{ExtCtxt, MacExpr, MacItem, MacResult};
use ext::build::AstBuilder;
use parse::token;

local_data_key!(registered_diagnostics: RefCell<HashMap<Name, Option<Name>>>)
local_data_key!(used_diagnostics: RefCell<HashMap<Name, Span>>)

fn with_registered_diagnostics<T>(f: |&mut HashMap<Name, Option<Name>>| -> T) -> T {
    match registered_diagnostics.get() {
        Some(cell) => f(cell.borrow_mut().deref_mut()),
        None => {
            let mut map = HashMap::new();
            let value = f(&mut map);
            registered_diagnostics.replace(Some(RefCell::new(map)));
            value
        }
    }
}

fn with_used_diagnostics<T>(f: |&mut HashMap<Name, Span>| -> T) -> T {
    match used_diagnostics.get() {
        Some(cell) => f(cell.borrow_mut().deref_mut()),
        None => {
            let mut map = HashMap::new();
            let value = f(&mut map);
            used_diagnostics.replace(Some(RefCell::new(map)));
            value
        }
    }
}

pub fn expand_diagnostic_used(ecx: &mut ExtCtxt, span: Span,
                              token_tree: &[TokenTree]) -> Box<MacResult> {
    let code = match token_tree {
        [ast::TTTok(_, token::IDENT(code, _))] => code,
        _ => unreachable!()
    };
    with_registered_diagnostics(|diagnostics| {
        if !diagnostics.contains_key(&code.name) {
            ecx.span_err(span, format!(
                "unknown diagnostic code {}", token::get_ident(code).get()
            ).as_slice());
        }
        ()
    });
    with_used_diagnostics(|diagnostics| {
        match diagnostics.swap(code.name, span) {
            Some(previous_span) => {
                ecx.span_warn(span, format!(
                    "diagnostic code {} already used", token::get_ident(code).get()
                ).as_slice());
                ecx.span_note(previous_span, "previous invocation");
            },
            None => ()
        }
        ()
    });
    MacExpr::new(quote_expr!(ecx, ()))
}

pub fn expand_register_diagnostic(ecx: &mut ExtCtxt, span: Span,
                                  token_tree: &[TokenTree]) -> Box<MacResult> {
    let (code, description) = match token_tree {
        [ast::TTTok(_, token::IDENT(ref code, _))] => {
            (code, None)
        },
        [ast::TTTok(_, token::IDENT(ref code, _)),
         ast::TTTok(_, token::COMMA),
         ast::TTTok(_, token::LIT_STR_RAW(description, _))] => {
            (code, Some(description))
        }
        _ => unreachable!()
    };
    with_registered_diagnostics(|diagnostics| {
        if !diagnostics.insert(code.name, description) {
            ecx.span_err(span, format!(
                "diagnostic code {} already registered", token::get_ident(*code).get()
            ).as_slice());
        }
    });
    let sym = Ident::new(token::gensym((
        "__register_diagnostic_".to_string() + token::get_ident(*code).get()
    ).as_slice()));
    MacItem::new(quote_item!(ecx, mod $sym {}).unwrap())
}

pub fn expand_build_diagnostic_array(ecx: &mut ExtCtxt, span: Span,
                                     token_tree: &[TokenTree]) -> Box<MacResult> {
    let name = match token_tree {
        [ast::TTTok(_, token::IDENT(ref name, _))] => name,
        _ => unreachable!()
    };

    let (count, expr) = with_used_diagnostics(|diagnostics_in_use| {
        with_registered_diagnostics(|diagnostics| {
            let descriptions: Vec<Gc<ast::Expr>> = diagnostics
                .iter().filter_map(|(code, description)| {
                if !diagnostics_in_use.contains_key(code) {
                    ecx.span_warn(span, format!(
                        "diagnostic code {} never used", token::get_name(*code).get()
                    ).as_slice());
                }
                description.map(|description| {
                    ecx.expr_tuple(span, vec![
                        ecx.expr_str(span, token::get_name(*code)),
                        ecx.expr_str(span, token::get_name(description))
                    ])
                })
            }).collect();
            (descriptions.len(), ecx.expr_vec(span, descriptions))
        })
    });
    MacItem::new(quote_item!(ecx,
        pub static $name: [(&'static str, &'static str), ..$count] = $expr;
    ).unwrap())
}
