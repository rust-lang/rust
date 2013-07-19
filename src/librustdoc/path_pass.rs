// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Records the full path to items


use astsrv;
use doc::ItemUtils;
use doc;
use fold::Fold;
use fold;
use pass::Pass;

#[cfg(test)] use extract;

use syntax::ast;

pub fn mk_pass() -> Pass {
    Pass {
        name: ~"path",
        f: run
    }
}

struct Ctxt {
    srv: astsrv::Srv,
    path: @mut ~[~str]
}

impl Clone for Ctxt {
    fn clone(&self) -> Ctxt {
        Ctxt {
            srv: self.srv.clone(),
            path: @mut (*self.path).clone()
        }
    }
}

fn run(srv: astsrv::Srv, doc: doc::Doc) -> doc::Doc {
    let ctxt = Ctxt {
        srv: srv,
        path: @mut ~[]
    };
    let fold = Fold {
        ctxt: ctxt.clone(),
        fold_item: fold_item,
        fold_mod: fold_mod,
        fold_nmod: fold_nmod,
        .. fold::default_any_fold(ctxt)
    };
    (fold.fold_doc)(&fold, doc)
}

fn fold_item(fold: &fold::Fold<Ctxt>, doc: doc::ItemDoc) -> doc::ItemDoc {
    doc::ItemDoc {
        path: (*fold.ctxt.path).clone(),
        .. doc
    }
}

fn fold_mod(fold: &fold::Fold<Ctxt>, doc: doc::ModDoc) -> doc::ModDoc {
    let is_topmod = doc.id() == ast::crate_node_id;

    if !is_topmod { fold.ctxt.path.push(doc.name()); }
    let doc = fold::default_any_fold_mod(fold, doc);
    if !is_topmod { fold.ctxt.path.pop(); }

    doc::ModDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        .. doc
    }
}

fn fold_nmod(fold: &fold::Fold<Ctxt>, doc: doc::NmodDoc) -> doc::NmodDoc {
    fold.ctxt.path.push(doc.name());
    let doc = fold::default_seq_fold_nmod(fold, doc);
    fold.ctxt.path.pop();

    doc::NmodDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        .. doc
    }
}

#[test]
fn should_record_mod_paths() {
    let source = ~"mod a { mod b { mod c { } } mod d { mod e { } } }";
    do astsrv::from_str(source) |srv| {
        let doc = extract::from_srv(srv.clone(), ~"");
        let doc = run(srv.clone(), doc);
        // hidden __std_macros module at the start.
        assert_eq!(doc.cratemod().mods()[1].mods()[0].mods()[0].path(),
                   ~[~"a", ~"b"]);
        assert_eq!(doc.cratemod().mods()[1].mods()[1].mods()[0].path(),
                   ~[~"a", ~"d"]);
    }
}

#[test]
fn should_record_fn_paths() {
    let source = ~"mod a { fn b() { } }";
    do astsrv::from_str(source) |srv| {
        let doc = extract::from_srv(srv.clone(), ~"");
        let doc = run(srv.clone(), doc);
        // hidden __std_macros module at the start.
        assert_eq!(doc.cratemod().mods()[1].fns()[0].path(), ~[~"a"]);
    }
}
