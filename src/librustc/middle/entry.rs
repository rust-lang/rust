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
use syntax::ast::{Crate, NodeId, item, item_fn};
use syntax::ast_map;
use syntax::attr;
use syntax::codemap::Span;
use syntax::parse::token::special_idents;
use syntax::visit;
use syntax::visit::Visitor;

struct EntryContext {
    session: Session,

    ast_map: ast_map::map,

    // The top-level function called 'main'
    main_fn: Option<(NodeId, Span)>,

    // The function that has attribute named 'main'
    attr_main_fn: Option<(NodeId, Span)>,

    // The function that has the attribute 'start' on it
    start_fn: Option<(NodeId, Span)>,

    // The functions that one might think are 'main' but aren't, e.g.
    // main functions not defined at the top level. For diagnostics.
    non_main_fns: ~[(NodeId, Span)],
}

impl Visitor<()> for EntryContext {
    fn visit_item(&mut self, item:@item, _:()) {
        find_item(item, self);
    }
}

pub fn find_entry_point(session: Session, crate: &Crate, ast_map: ast_map::map) {
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

fn find_item(item: @item, ctxt: &mut EntryContext) {
    match item.node {
        item_fn(..) => {
            if item.ident.name == special_idents::main.name {
                match ctxt.ast_map.find(&item.id) {
                    Some(&ast_map::node_item(_, path)) => {
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
