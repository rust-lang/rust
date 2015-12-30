// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use front::map as ast_map;
use middle::def_id::{CRATE_DEF_INDEX};
use session::{config, Session};
use syntax::ast::NodeId;
use syntax::attr;
use syntax::codemap::Span;
use syntax::entry::EntryPointType;
use rustc_front::hir::{Item, ItemFn};
use rustc_front::intravisit::Visitor;

struct EntryContext<'a, 'tcx: 'a> {
    session: &'a Session,

    map: &'a ast_map::Map<'tcx>,

    // The top-level function called 'main'
    main_fn: Option<(NodeId, Span)>,

    // The function that has attribute named 'main'
    attr_main_fn: Option<(NodeId, Span)>,

    // The function that has the attribute 'start' on it
    start_fn: Option<(NodeId, Span)>,

    // The functions that one might think are 'main' but aren't, e.g.
    // main functions not defined at the top level. For diagnostics.
    non_main_fns: Vec<(NodeId, Span)> ,
}

impl<'a, 'tcx> Visitor<'tcx> for EntryContext<'a, 'tcx> {
    fn visit_item(&mut self, item: &'tcx Item) {
        let def_id = self.map.local_def_id(item.id);
        let def_key = self.map.def_key(def_id);
        let at_root = def_key.parent == Some(CRATE_DEF_INDEX);
        find_item(item, self, at_root);
    }
}

pub fn find_entry_point(session: &Session, ast_map: &ast_map::Map) {
    let any_exe = session.crate_types.borrow().iter().any(|ty| {
        *ty == config::CrateTypeExecutable
    });
    if !any_exe {
        // No need to find a main function
        return
    }

    // If the user wants no main function at all, then stop here.
    if attr::contains_name(&ast_map.krate().attrs, "no_main") {
        session.entry_type.set(Some(config::EntryNone));
        return
    }

    let mut ctxt = EntryContext {
        session: session,
        map: ast_map,
        main_fn: None,
        attr_main_fn: None,
        start_fn: None,
        non_main_fns: Vec::new(),
    };

    ast_map.krate().visit_all_items(&mut ctxt);

    configure_main(&mut ctxt);
}

// Beware, this is duplicated in libsyntax/entry.rs, make sure to keep
// them in sync.
fn entry_point_type(item: &Item, at_root: bool) -> EntryPointType {
    match item.node {
        ItemFn(..) => {
            if attr::contains_name(&item.attrs, "start") {
                EntryPointType::Start
            } else if attr::contains_name(&item.attrs, "main") {
                EntryPointType::MainAttr
            } else if item.name.as_str() == "main" {
                if at_root {
                    // This is a top-level function so can be 'main'
                    EntryPointType::MainNamed
                } else {
                    EntryPointType::OtherMain
                }
            } else {
                EntryPointType::None
            }
        }
        _ => EntryPointType::None,
    }
}


fn find_item(item: &Item, ctxt: &mut EntryContext, at_root: bool) {
    match entry_point_type(item, at_root) {
        EntryPointType::MainNamed => {
            if ctxt.main_fn.is_none() {
                ctxt.main_fn = Some((item.id, item.span));
            } else {
                span_err!(ctxt.session, item.span, E0136,
                          "multiple 'main' functions");
            }
        },
        EntryPointType::OtherMain => {
            ctxt.non_main_fns.push((item.id, item.span));
        },
        EntryPointType::MainAttr => {
            if ctxt.attr_main_fn.is_none() {
                ctxt.attr_main_fn = Some((item.id, item.span));
            } else {
                span_err!(ctxt.session, item.span, E0137,
                          "multiple functions with a #[main] attribute");
            }
        },
        EntryPointType::Start => {
            if ctxt.start_fn.is_none() {
                ctxt.start_fn = Some((item.id, item.span));
            } else {
                span_err!(ctxt.session, item.span, E0138,
                          "multiple 'start' functions");
            }
        },
        EntryPointType::None => ()
    }
}

fn configure_main(this: &mut EntryContext) {
    if this.start_fn.is_some() {
        *this.session.entry_fn.borrow_mut() = this.start_fn;
        this.session.entry_type.set(Some(config::EntryStart));
    } else if this.attr_main_fn.is_some() {
        *this.session.entry_fn.borrow_mut() = this.attr_main_fn;
        this.session.entry_type.set(Some(config::EntryMain));
    } else if this.main_fn.is_some() {
        *this.session.entry_fn.borrow_mut() = this.main_fn;
        this.session.entry_type.set(Some(config::EntryMain));
    } else {
        // No main function
        let mut err = this.session.struct_err("main function not found");
        if !this.non_main_fns.is_empty() {
            // There were some functions named 'main' though. Try to give the user a hint.
            err.note("the main function must be defined at the crate level \
                      but you have one or more functions named 'main' that are not \
                      defined at the crate level. Either move the definition or \
                      attach the `#[main]` attribute to override this behavior.");
            for &(_, span) in &this.non_main_fns {
                err.span_note(span, "here is a function named 'main'");
            }
            err.emit();
            this.session.abort_if_errors();
        } else {
            err.emit();
        }
    }
}
