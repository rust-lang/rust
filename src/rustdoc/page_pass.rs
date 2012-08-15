/*!
 * Divides the document tree into pages.
 *
 * Each page corresponds is a logical section. There may be pages for
 * individual modules, pages for the crate, indexes, etc.
 */

import doc::{item_utils, page_utils};
import syntax::ast;

export mk_pass;

fn mk_pass(output_style: config::output_style) -> pass {
    {
        name: ~"page",
        f: fn~(srv: astsrv::srv, doc: doc::doc) -> doc::doc {
            run(srv, doc, output_style)
        }
    }
}

fn run(
    _srv: astsrv::srv,
    doc: doc::doc,
    output_style: config::output_style
) -> doc::doc {

    if output_style == config::doc_per_crate {
        return doc;
    }

    let (result_port, page_chan) = do task::spawn_conversation
        |page_port, result_chan| {
        comm::send(result_chan, make_doc_from_pages(page_port));
    };

    find_pages(doc, page_chan);
    comm::recv(result_port)
}

type page_port = comm::Port<option<doc::page>>;
type page_chan = comm::Chan<option<doc::page>>;

fn make_doc_from_pages(page_port: page_port) -> doc::doc {
    let mut pages = ~[];
    loop {
        let val = comm::recv(page_port);
        if option::is_some(val) {
            pages += ~[option::unwrap(val)];
        } else {
            break;
        }
    }
    doc::doc_({
        pages: pages
    })
}

fn find_pages(doc: doc::doc, page_chan: page_chan) {
    let fold = fold::fold({
        fold_crate: fold_crate,
        fold_mod: fold_mod,
        fold_nmod: fold_nmod
        with *fold::default_any_fold(page_chan)
    });
    fold.fold_doc(fold, doc);

    comm::send(page_chan, none);
}

fn fold_crate(
    fold: fold::fold<page_chan>,
    doc: doc::cratedoc
) -> doc::cratedoc {

    let doc = fold::default_seq_fold_crate(fold, doc);

    let page = doc::cratepage({
        topmod: strip_mod(doc.topmod)
        with doc
    });

    comm::send(fold.ctxt, some(page));

    doc
}

fn fold_mod(
    fold: fold::fold<page_chan>,
    doc: doc::moddoc
) -> doc::moddoc {

    let doc = fold::default_any_fold_mod(fold, doc);

    if doc.id() != ast::crate_node_id {

        let doc = strip_mod(doc);
        let page = doc::itempage(doc::modtag(doc));
        comm::send(fold.ctxt, some(page));
    }

    doc
}

fn strip_mod(doc: doc::moddoc) -> doc::moddoc {
    doc::moddoc_({
        items: do vec::filter(doc.items) |item| {
            match item {
              doc::modtag(_) => false,
              doc::nmodtag(_) => false,
              _ => true
            }
        }
        with *doc
    })
}

fn fold_nmod(
    fold: fold::fold<page_chan>,
    doc: doc::nmoddoc
) -> doc::nmoddoc {
    let doc = fold::default_seq_fold_nmod(fold, doc);
    let page = doc::itempage(doc::nmodtag(doc));
    comm::send(fold.ctxt, some(page));
    return doc;
}

#[test]
fn should_not_split_the_doc_into_pages_for_doc_per_crate() {
    let doc = test::mk_doc_(
        config::doc_per_crate,
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
    fn mk_doc_(
        output_style: config::output_style,
        source: ~str
    ) -> doc::doc {
        do astsrv::from_str(source) |srv| {
            let doc = extract::from_srv(srv, ~"");
            run(srv, doc, output_style)
        }
    }

    fn mk_doc(source: ~str) -> doc::doc {
        mk_doc_(config::doc_per_mod, source)
    }
}
