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
use ast;
use ast::{Ident, Name, TokenTree};
use codemap::Span;
use ext::base::{ExtCtxt, MacExpr, MacResult, MacItems};
use ext::build::AstBuilder;
use parse::token;
use ptr::P;

thread_local! {
    static REGISTERED_DIAGNOSTICS: RefCell<HashMap<Name, Option<Name>>> = {
        RefCell::new(HashMap::new())
    }
}
thread_local! {
    static USED_DIAGNOSTICS: RefCell<HashMap<Name, Span>> = {
        RefCell::new(HashMap::new())
    }
}

fn with_registered_diagnostics<T, F>(f: F) -> T where
    F: FnOnce(&mut HashMap<Name, Option<Name>>) -> T,
{
    REGISTERED_DIAGNOSTICS.with(move |slot| {
        f(&mut *slot.borrow_mut())
    })
}

fn with_used_diagnostics<T, F>(f: F) -> T where
    F: FnOnce(&mut HashMap<Name, Span>) -> T,
{
    USED_DIAGNOSTICS.with(move |slot| {
        f(&mut *slot.borrow_mut())
    })
}

pub fn expand_diagnostic_used<'cx>(ecx: &'cx mut ExtCtxt,
                                   span: Span,
                                   token_tree: &[TokenTree])
                                   -> Box<MacResult+'cx> {
    let code = match token_tree {
        [ast::TtToken(_, token::Ident(code, _))] => code,
        _ => unreachable!()
    };
    with_used_diagnostics(|diagnostics| {
        match diagnostics.insert(code.name, span) {
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

pub fn expand_register_diagnostic<'cx>(ecx: &'cx mut ExtCtxt,
                                       span: Span,
                                       token_tree: &[TokenTree])
                                       -> Box<MacResult+'cx> {
    let (code, description) = match token_tree {
        [ast::TtToken(_, token::Ident(ref code, _))] => {
            (code, None)
        },
        [ast::TtToken(_, token::Ident(ref code, _)),
         ast::TtToken(_, token::Comma),
         ast::TtToken(_, token::Literal(token::StrRaw(description, _), None))] => {
            (code, Some(description))
        }
        _ => unreachable!()
    };
    with_registered_diagnostics(|diagnostics| {
        if diagnostics.insert(code.name, description).is_some() {
            ecx.span_err(span, format!(
                "diagnostic code {} already registered", token::get_ident(*code).get()
            ).as_slice());
        }
    });
    let sym = Ident::new(token::gensym((
        "__register_diagnostic_".to_string() + token::get_ident(*code).get()
    ).as_slice()));
    MacItems::new(vec![quote_item!(ecx, mod $sym {}).unwrap()].into_iter())
}

pub fn expand_build_diagnostic_array<'cx>(ecx: &'cx mut ExtCtxt,
                                          span: Span,
                                          token_tree: &[TokenTree])
                                          -> Box<MacResult+'cx> {
    let name = match token_tree {
        [ast::TtToken(_, token::Ident(ref name, _))] => name,
        _ => unreachable!()
    };

    let (count, expr) =
        with_registered_diagnostics(|diagnostics| {
            let descriptions: Vec<P<ast::Expr>> =
                diagnostics.iter().filter_map(|(code, description)| {
                    description.map(|description| {
                        ecx.expr_tuple(span, vec![
                            ecx.expr_str(span, token::get_name(*code)),
                            ecx.expr_str(span, token::get_name(description))])
                    })
                }).collect();
            (descriptions.len(), ecx.expr_vec(span, descriptions))
        });

    MacItems::new(vec![quote_item!(ecx,
        pub static $name: [(&'static str, &'static str), ..$count] = $expr;
    ).unwrap()].into_iter())
}
