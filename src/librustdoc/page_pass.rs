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
Divides the document tree into pages.

Each page corresponds is a logical section. There may be pages for
individual modules, pages for the crate, indexes, etc.
*/

use doc::{ItemUtils, PageUtils};
use syntax::ast;
use util::NominalOp;

pub fn mk_pass(output_style: config::OutputStyle) -> Pass {
    {
        name: ~"page",
        f: fn~(srv: astsrv::Srv, +doc: doc::Doc) -> doc::Doc {
            run(srv, doc, output_style)
        }
    }
}

fn run(
    _srv: astsrv::Srv,
    +doc: doc::Doc,
    output_style: config::OutputStyle
) -> doc::Doc {

    if output_style == config::DocPerCrate {
        return doc;
    }

    let (result_port, page_chan) = do task::spawn_conversation
        |page_port, result_chan| {
        comm::send(result_chan, make_doc_from_pages(page_port));
    };

    find_pages(doc, page_chan);
    comm::recv(result_port)
}

type PagePort = comm::Port<Option<doc::Page>>;
type PageChan = comm::Chan<Option<doc::Page>>;

type NominalPageChan = NominalOp<PageChan>;

fn make_doc_from_pages(page_port: PagePort) -> doc::Doc {
    let mut pages = ~[];
    loop {
        let val = comm::recv(page_port);
        if val.is_some() {
            pages += ~[option::unwrap(move val)];
        } else {
            break;
        }
    }
    doc::Doc_({
        pages: pages
    })
}

fn find_pages(doc: doc::Doc, page_chan: PageChan) {
    let fold = fold::Fold({
        fold_crate: fold_crate,
        fold_mod: fold_mod,
        fold_nmod: fold_nmod,
        .. *fold::default_any_fold(NominalOp { op: page_chan })
    });
    (fold.fold_doc)(&fold, doc);

    comm::send(page_chan, None);
}

fn fold_crate(
    fold: &fold::Fold<NominalPageChan>,
    +doc: doc::CrateDoc
) -> doc::CrateDoc {

    let doc = fold::default_seq_fold_crate(fold, doc);

    let page = doc::CratePage({
        topmod: strip_mod(doc.topmod),
        .. doc
    });

    comm::send(fold.ctxt.op, Some(page));

    doc
}

fn fold_mod(
    fold: &fold::Fold<NominalPageChan>,
    +doc: doc::ModDoc
) -> doc::ModDoc {

    let doc = fold::default_any_fold_mod(fold, doc);

    if doc.id() != ast::crate_node_id {

        let doc = strip_mod(doc);
        let page = doc::ItemPage(doc::ModTag(doc));
        comm::send(fold.ctxt.op, Some(page));
    }

    doc
}

fn strip_mod(doc: doc::ModDoc) -> doc::ModDoc {
    doc::ModDoc_({
        items: do vec::filter(doc.items) |item| {
            match *item {
              doc::ModTag(_) => false,
              doc::NmodTag(_) => false,
              _ => true
            }
        },
        .. *doc
    })
}

fn fold_nmod(
    fold: &fold::Fold<NominalPageChan>,
    +doc: doc::NmodDoc
) -> doc::NmodDoc {
    let doc = fold::default_seq_fold_nmod(fold, doc);
    let page = doc::ItemPage(doc::NmodTag(doc));
    comm::send(fold.ctxt.op, Some(page));
    return doc;
}

#[test]
fn should_not_split_the_doc_into_pages_for_doc_per_crate() {
    let doc = test::mk_doc_(
        config::DocPerCrate,
        ~"mod a { } mod b { mod c { } }"
    );
    assert doc.pages.len() == 1u;
}

#[test]
fn should_make_a_page_for_every_mod() {
    let doc = test::mk_doc(~"mod a { }");
    assert doc.pages.mods()[0].name() == ~"a";
}

#[test]
fn should_remove_mods_from_containing_mods() {
    let doc = test::mk_doc(~"mod a { }");
    assert vec::is_empty(doc.cratemod().mods());
}

#[test]
fn should_make_a_page_for_every_foreign_mod() {
    let doc = test::mk_doc(~"extern mod a { }");
    assert doc.pages.nmods()[0].name() == ~"a";
}

#[test]
fn should_remove_foreign_mods_from_containing_mods() {
    let doc = test::mk_doc(~"extern mod a { }");
    assert vec::is_empty(doc.cratemod().nmods());
}

#[cfg(test)]
mod test {
    #[legacy_exports];
    fn mk_doc_(
        output_style: config::OutputStyle,
        source: ~str
    ) -> doc::Doc {
        do astsrv::from_str(source) |srv| {
            let doc = extract::from_srv(srv, ~"");
            run(srv, doc, output_style)
        }
    }

    fn mk_doc(source: ~str) -> doc::Doc {
        mk_doc_(config::DocPerMod, source)
    }
}
