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

struct EntryContext {
    session: Session,

    // The function that has attribute named 'main'
    attr_main_fn: Option<(node_id, span)>,

    // The functions that could be main functions
    main_fns: ~[Option<(node_id, span)>],

    // The function that has the attribute 'start' on it
    start_fn: Option<(node_id, span)>,
}

type EntryVisitor = vt<@mut EntryContext>;

pub fn find_entry_point(session: Session, crate: @crate) {

    let ctxt = @mut EntryContext {
        session: session,
        attr_main_fn: None,
        main_fns: ~[],
        start_fn: None,
    };

    visit_crate(crate, ctxt, mk_vt(@Visitor {
        visit_item: |item, ctxt, visitor| find_item(item, ctxt, visitor),
        .. *default_visitor()
    }));

    check_duplicate_main(ctxt);
}

fn find_item(item: @item, ctxt: @mut EntryContext, visitor: EntryVisitor) {
    match item.node {
        item_fn(*) => {
            // If this is the main function, we must record it in the
            // session.

            // FIXME #4404 android JNI hacks
            if !*ctxt.session.building_library ||
                ctxt.session.targ_cfg.os == session::os_android {

                if ctxt.attr_main_fn.is_none() &&
                    item.ident == special_idents::main {

                    ctxt.main_fns.push(Some((item.id, item.span)));
                }

                if attrs_contains_name(item.attrs, ~"main") {
                    if ctxt.attr_main_fn.is_none() {
                        ctxt.attr_main_fn = Some((item.id, item.span));
                    } else {
                        ctxt.session.span_err(
                            item.span,
                            ~"multiple 'main' functions");
                    }
                }

                if attrs_contains_name(item.attrs, ~"start") {
                    if ctxt.start_fn.is_none() {
                        ctxt.start_fn = Some((item.id, item.span));
                    } else {
                        ctxt.session.span_err(
                            item.span,
                            ~"multiple 'start' functions");
                    }
                }
            }
        }
        _ => ()
    }

    visit_item(item, ctxt, visitor);
}

// main function checking
//
// be sure that there is only one main function
fn check_duplicate_main(ctxt: @mut EntryContext) {
    let this = &mut *ctxt;
    if this.attr_main_fn.is_none() && this.start_fn.is_none() {
        if this.main_fns.len() >= 1u {
            let mut i = 1u;
            while i < this.main_fns.len() {
                let (_, dup_main_span) = this.main_fns[i].unwrap();
                this.session.span_err(
                    dup_main_span,
                    ~"multiple 'main' functions");
                i += 1;
            }
            *this.session.entry_fn = this.main_fns[0];
            *this.session.entry_type = Some(session::EntryMain);
        }
    } else if !this.start_fn.is_none() {
        *this.session.entry_fn = this.start_fn;
        *this.session.entry_type = Some(session::EntryStart);
    } else {
        *this.session.entry_fn = this.attr_main_fn;
        *this.session.entry_type = Some(session::EntryMain);
    }
}
