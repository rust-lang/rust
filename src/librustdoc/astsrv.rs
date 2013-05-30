// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
Provides all access to AST-related, non-sendable info

Rustdoc is intended to be parallel, and the rustc AST is filled with
shared boxes. The AST service attempts to provide a single place to
query AST-related information, shielding the rest of Rustdoc from its
non-sendableness.
*/

use core::prelude::*;

use parse;

use core::cell::Cell;
use core::comm::{stream, SharedChan, Port};
use rustc::driver::driver;
use rustc::driver::session::Session;
use rustc::driver::session::{basic_options, options};
use rustc::front;
use syntax::ast;
use syntax::ast_map;
use syntax;

pub struct Ctxt {
    ast: @ast::crate,
    ast_map: ast_map::map
}

type SrvOwner<'self,T> = &'self fn(srv: Srv) -> T;
pub type CtxtHandler<T> = ~fn(ctxt: Ctxt) -> T;
type Parser = ~fn(Session, s: ~str) -> @ast::crate;

enum Msg {
    HandleRequest(~fn(Ctxt)),
    Exit
}

#[deriving(Clone)]
pub struct Srv {
    ch: SharedChan<Msg>
}

pub fn from_str<T>(source: ~str, owner: SrvOwner<T>) -> T {
    run(owner, copy source, parse::from_str_sess)
}

pub fn from_file<T>(file: ~str, owner: SrvOwner<T>) -> T {
    run(owner, copy file, |sess, f| parse::from_file_sess(sess, &Path(f)))
}

fn run<T>(owner: SrvOwner<T>, source: ~str, parse: Parser) -> T {

    let (po, ch) = stream();

    let source = Cell(source);
    let parse = Cell(parse);
    do task::spawn {
        act(&po, source.take(), parse.take());
    }

    let srv_ = Srv {
        ch: SharedChan::new(ch)
    };

    let res = owner(srv_.clone());
    srv_.ch.send(Exit);
    res
}

fn act(po: &Port<Msg>, source: ~str, parse: Parser) {
    let sess = build_session();

    let ctxt = build_ctxt(
        sess,
        parse(sess, copy source)
    );

    let mut keep_going = true;
    while keep_going {
        match po.recv() {
          HandleRequest(f) => {
            f(ctxt);
          }
          Exit => {
            keep_going = false;
          }
        }
    }
}

pub fn exec<T:Owned>(
    srv: Srv,
    f: ~fn(ctxt: Ctxt) -> T
) -> T {
    let (po, ch) = stream();
    let msg = HandleRequest(|ctxt| ch.send(f(ctxt)));
    srv.ch.send(msg);
    po.recv()
}

fn build_ctxt(sess: Session,
              ast: @ast::crate) -> Ctxt {

    use rustc::front::config;

    let ast = config::strip_unconfigured_items(ast);
    let ast = syntax::ext::expand::expand_crate(sess.parse_sess,
                                                copy sess.opts.cfg, ast);
    let ast = front::test::modify_for_testing(sess, ast);
    let ast_map = ast_map::map_crate(sess.diagnostic(), ast);

    Ctxt {
        ast: ast,
        ast_map: ast_map,
    }
}

fn build_session() -> Session {
    let sopts: @options = basic_options();
    let emitter = syntax::diagnostic::emit;

    let session = driver::build_session(sopts, emitter);
    session
}

#[test]
fn should_prune_unconfigured_items() {
    let source = ~"#[cfg(shut_up_and_leave_me_alone)]fn a() { }";
    do from_str(source) |srv| {
        do exec(srv) |ctxt| {
            assert!(vec::is_empty(ctxt.ast.node.module.items));
        }
    }
}

#[test]
fn srv_should_build_ast_map() {
    let source = ~"fn a() { }";
    do from_str(source) |srv| {
        do exec(srv) |ctxt| {
            assert!(!ctxt.ast_map.is_empty())
        };
    }
}

#[test]
fn should_ignore_external_import_paths_that_dont_exist() {
    let source = ~"use forble; use forble::bippy;";
    from_str(source, |_srv| { } )
}

#[test]
fn srv_should_return_request_result() {
    let source = ~"fn a() { }";
    do from_str(source) |srv| {
        let result = exec(srv, |_ctxt| 1000 );
        assert_eq!(result, 1000);
    }
}
