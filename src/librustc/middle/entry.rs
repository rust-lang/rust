// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use driver::session;
use driver::session::Session;
use metadata::csearch;

use syntax::ast;
use syntax::ast_map;
use syntax::attr;
use syntax::codemap::Span;
use syntax::parse::token::special_idents;
use syntax::visit;
use syntax::visit::Visitor;

struct EntryContext {
    session: Session,

    ast_map: ast_map::Map,

    // The top-level function called 'main'
    main_fn: Option<(ast::NodeId, Span)>,

    // The function that has attribute named 'main'
    attr_main_fn: Option<(ast::NodeId, Span)>,

    // The function that has the attribute 'start' on it
    start_fn: Option<(ast::NodeId, Span)>,

    // The functions that one might think are 'main' but aren't, e.g.
    // main functions not defined at the top level. For diagnostics.
    non_main_fns: ~[(ast::NodeId, Span)],
}

impl Visitor<()> for EntryContext {
    fn visit_item(&mut self, item: &ast::Item, _:()) {
        find_item(item, self);
    }
}

pub fn find_entry_point(session: Session, crate: &ast::Crate, ast_map: ast_map::Map) {
    if session.building_library.get() {
        // No need to find a main function
        return;
    }

    // If the user wants no main function at all, then stop here.
    if attr::contains_name(crate.attrs, "no_main") {
        session.entry_type.set(Some(session::EntryNone));
        return
    }

    let mut ctxt = EntryContext {
        session: session,
        ast_map: ast_map,
        main_fn: None,
        attr_main_fn: None,
        start_fn: None,
        non_main_fns: ~[],
    };

    visit::walk_crate(&mut ctxt, crate, ());

    configure_main(&mut ctxt);
}

fn find_item(item: &ast::Item, ctxt: &mut EntryContext) {
    match item.node {
        ast::ItemFn(..) => {
            if item.ident.name == special_idents::main.name {
                {
                    let ast_map = ctxt.ast_map.borrow();
                    match ast_map.get().find(&item.id) {
                        Some(&ast_map::NodeItem(_, path)) => {
                            if path.len() == 0 {
                                // This is a top-level function so can be 'main'
                                if ctxt.main_fn.is_none() {
                                    ctxt.main_fn = Some((item.id, item.span));
                                } else {
                                    ctxt.session.span_err(
                                        item.span,
                                        "multiple 'main' functions");
                                }
                            } else {
                                // This isn't main
                                ctxt.non_main_fns.push((item.id, item.span));
                            }
                        }
                        _ => unreachable!()
                    }
                }
            }

            if attr::contains_name(item.attrs, "main") {
                if ctxt.attr_main_fn.is_none() {
                    ctxt.attr_main_fn = Some((item.id, item.span));
                } else {
                    ctxt.session.span_err(
                        item.span,
                        "multiple 'main' functions");
                }
            }

            if attr::contains_name(item.attrs, "start") {
                if ctxt.start_fn.is_none() {
                    ctxt.start_fn = Some((item.id, item.span));
                } else {
                    ctxt.session.span_err(
                        item.span,
                        "multiple 'start' functions");
                }
            }
        }
        _ => ()
    }

    visit::walk_item(ctxt, item, ());
}

fn configure_main(this: &mut EntryContext) {
    if this.start_fn.is_some() {
        this.session.entry_fn.set(this.start_fn);
        this.session.entry_type.set(Some(session::EntryStart));
    } else if this.attr_main_fn.is_some() {
        this.session.entry_fn.set(this.attr_main_fn);
        this.session.entry_type.set(Some(session::EntryMain));
    } else if this.main_fn.is_some() {
        this.session.entry_fn.set(this.main_fn);
        this.session.entry_type.set(Some(session::EntryMain));
    } else {
        if !this.session.building_library.get() {
            // No main function
            this.session.err("main function not found");
            if !this.non_main_fns.is_empty() {
                // There were some functions named 'main' though. Try to give the user a hint.
                this.session.note("the main function must be defined at the crate level \
                                   but you have one or more functions named 'main' that are not \
                                   defined at the crate level. Either move the definition or \
                                   attach the `#[main]` attribute to override this behavior.");
                for &(_, span) in this.non_main_fns.iter() {
                    this.session.span_note(span, "here is a function named 'main'");
                }
            }
            this.session.abort_if_errors();
        }
    }
}

struct BootFinder {
    local_candidates: ~[(ast::NodeId, Span)],
    extern_candidates: ~[(ast::CrateNum, Span)],
    sess: Session,
}

pub fn find_boot_fn(sess: Session, crate: &ast::Crate) {
    let mut cx = BootFinder {
        local_candidates: ~[],
        extern_candidates: ~[],
        sess: sess,
    };
    visit::walk_crate(&mut cx, crate, ());
    match cx.local_candidates.len() + cx.extern_candidates.len() {
        0 => {
            if !sess.building_library.get() {
                match sess.entry_type.get() {
                    Some(session::EntryNone) => {},
                    Some(session::EntryStart) => {},
                    Some(session::EntryMain) => {
                        sess.err("main function found but no #[boot] \
                                  functions found");
                    }
                    None => {
                        sess.bug("expected entry calculation by now");
                    }
                }
            }
        }

        1 => {
            for &(id, _) in cx.local_candidates.iter() {
                sess.boot_fn.set(Some(ast::DefId {
                    crate: ast::LOCAL_CRATE,
                    node: id,
                }));
            }
            for &(cnum, span) in cx.extern_candidates.iter() {
                sess.boot_fn.set(csearch::get_boot_fn(sess.cstore, cnum));
                if sess.boot_fn.get().is_none() {
                    sess.span_err(span, "no #[boot] function found in crate");
                }
            }
        }

        _ => {
            sess.err("too many #[boot] functions found");
            let lcandidates = cx.local_candidates.iter().map(|&(_, span)| span);
            let ecandidates = cx.extern_candidates.iter().map(|&(_, span)| span);
            for (i, span) in ecandidates.chain(lcandidates).enumerate() {
                sess.span_note(span, format!(r"candidate \#{}", i));
            }
        }
    }

    sess.abort_if_errors();
}

impl Visitor<()> for BootFinder {
    fn visit_item(&mut self, it: @ast::item, _: ()) {
        match it.node {
            ast::item_fn(..) if attr::contains_name(it.attrs, "boot") => {
                self.local_candidates.push((it.id, it.span))
            }
            _ => {}
        }
        visit::walk_item(self, it, ());
    }

    fn visit_view_item(&mut self, it: &ast::view_item, _: ()) {
        match it.node {
            ast::view_item_extern_mod(name, _, _, _) => {
                if attr::contains_name(it.attrs, "boot") {
                    self.sess.cstore.iter_crate_data(|num, meta| {
                        if meta.name == self.sess.str_of(name) {
                            self.extern_candidates.push((num, it.span));
                        }
                    });
                }
            }
            _ => {}
        }
        visit::walk_view_item(self, it, ());
    }
}
