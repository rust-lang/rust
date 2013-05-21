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
use syntax::parse::token::special_idents;
use syntax::ast::{crate, node_id, item, item_fn};
use syntax::codemap::span;
use syntax::visit::{default_visitor, mk_vt, vt, Visitor, visit_crate, visit_item};
use syntax::attr::{attrs_contains_name};
use syntax::ast_map;
use core::util;

struct EntryContext {
    session: Session,

    ast_map: ast_map::map,

    // The top-level function called 'main'
    main_fn: Option<(node_id, span)>,

    // The function that has attribute named 'main'
    attr_main_fn: Option<(node_id, span)>,

    // The function that has the attribute 'start' on it
    start_fn: Option<(node_id, span)>,

    // The functions that one might think are 'main' but aren't, e.g.
    // main functions not defined at the top level. For diagnostics.
    non_main_fns: ~[(node_id, span)],
}

type EntryVisitor = vt<@mut EntryContext>;

pub fn find_entry_point(session: Session, crate: @crate, ast_map: ast_map::map) {

    // FIXME #4404 android JNI hacks
    if *session.building_library &&
        session.targ_cfg.os != session::os_android {
        // No need to find a main function
        return;
    }

    let ctxt = @mut EntryContext {
        session: session,
        ast_map: ast_map,
        main_fn: None,
        attr_main_fn: None,
        start_fn: None,
        non_main_fns: ~[],
    };

    visit_crate(crate, ctxt, mk_vt(@Visitor {
        visit_item: |item, ctxt, visitor| find_item(item, ctxt, visitor),
        .. *default_visitor()
    }));

    configure_main(ctxt);
}

fn find_item(item: @item, ctxt: @mut EntryContext, visitor: EntryVisitor) {
    match item.node {
        item_fn(*) => {
            if item.ident == special_idents::main {
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
                    _ => util::unreachable()
                }
            }

            if attrs_contains_name(item.attrs, "main") {
                if ctxt.attr_main_fn.is_none() {
                    ctxt.attr_main_fn = Some((item.id, item.span));
                } else {
                    ctxt.session.span_err(
                        item.span,
                        "multiple 'main' functions");
                }
            }

            if attrs_contains_name(item.attrs, "start") {
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

    visit_item(item, ctxt, visitor);
}

fn configure_main(ctxt: @mut EntryContext) {
    let this = &mut *ctxt;
    if this.start_fn.is_some() {
        *this.session.entry_fn = this.start_fn;
        *this.session.entry_type = Some(session::EntryStart);
    } else if this.attr_main_fn.is_some() {
        *this.session.entry_fn = this.attr_main_fn;
        *this.session.entry_type = Some(session::EntryMain);
    } else if this.main_fn.is_some() {
        *this.session.entry_fn = this.main_fn;
        *this.session.entry_type = Some(session::EntryMain);
    } else {
        if !*this.session.building_library {
            // No main function
            this.session.err("main function not found");
            if !this.non_main_fns.is_empty() {
                // There were some functions named 'main' though. Try to give the user a hint.
                this.session.note("the main function must be defined at the crate level \
                                   but you have one or more functions named 'main' that are not \
                                   defined at the crate level. Either move the definition or \
                                   attach the `#[main]` attribute to override this behavior.");
                for this.non_main_fns.each |&(_, span)| {
                    this.session.span_note(span, "here is a function named 'main'");
                }
            }
            this.session.abort_if_errors();
        } else {
            // If we *are* building a library, then we're on android where we still might
            // optionally want to translate main $4404
            assert_eq!(this.session.targ_cfg.os, session::os_android);
        }
    }
}
