// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An experimental pass that scources for `#[rustc_mir]` attributes,
//! builds the resulting MIR, and dumps it out into a file for inspection.
//!
//! The attribute formats that are currently accepted are:
//!
//! - `#[rustc_mir(graphviz="file.gv")]`
//! - `#[rustc_mir(pretty="file.mir")]`

use build;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::mir::Mir;
use rustc::mir::transform::{MirSource, MIR_CONST, MIR_VALIDATED, MIR_OPTIMIZED};
use rustc::ty::{self, TyCtxt};
use rustc::ty::maps::Providers;
use rustc::ty::steal::Steal;
use rustc::hir;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::util::nodemap::DefIdSet;
use syntax::ast;
use syntax_pos::{DUMMY_SP, Span};
use transform;

use std::rc::Rc;

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        mir_keys,
        mir_const,
        mir_validated,
        optimized_mir,
        is_mir_available,
        ..*providers
    };
}

fn is_mir_available<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> bool {
    tcx.mir_keys(def_id.krate).contains(&def_id)
}

/// Finds the full set of def-ids within the current crate that have
/// MIR associated with them.
fn mir_keys<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, krate: CrateNum)
                      -> Rc<DefIdSet> {
    assert_eq!(krate, LOCAL_CRATE);

    let mut set = DefIdSet();

    // All body-owners have MIR associated with them.
    set.extend(tcx.body_owners());

    // Additionally, tuple struct/variant constructors have MIR, but
    // they don't have a BodyId, so we need to build them separately.
    struct GatherCtors<'a, 'tcx: 'a> {
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        set: &'a mut DefIdSet,
    }
    impl<'a, 'tcx> Visitor<'tcx> for GatherCtors<'a, 'tcx> {
        fn visit_variant_data(&mut self,
                              v: &'tcx hir::VariantData,
                              _: ast::Name,
                              _: &'tcx hir::Generics,
                              _: ast::NodeId,
                              _: Span) {
            if let hir::VariantData::Tuple(_, node_id) = *v {
                self.set.insert(self.tcx.hir.local_def_id(node_id));
            }
            intravisit::walk_struct_def(self, v)
        }
        fn nested_visit_map<'b>(&'b mut self) -> NestedVisitorMap<'b, 'tcx> {
            NestedVisitorMap::None
        }
    }
    tcx.hir.krate().visit_all_item_likes(&mut GatherCtors {
        tcx: tcx,
        set: &mut set,
    }.as_deep_visitor());

    Rc::new(set)
}

fn mir_const<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Steal<Mir<'tcx>> {
    let mut mir = build::mir_build(tcx, def_id);
    let source = MirSource::from_local_def_id(tcx, def_id);
    transform::run_suite(tcx, source, MIR_CONST, &mut mir);
    tcx.alloc_steal_mir(mir)
}

fn mir_validated<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Steal<Mir<'tcx>> {
    let source = MirSource::from_local_def_id(tcx, def_id);
    if let MirSource::Const(_) = source {
        // Ensure that we compute the `mir_const_qualif` for constants at
        // this point, before we steal the mir-const result. We don't
        // directly need the result or `mir_const_qualif`, so we can just force it.
        ty::queries::mir_const_qualif::force(tcx, DUMMY_SP, def_id);
    }

    let mut mir = tcx.mir_const(def_id).steal();
    transform::run_suite(tcx, source, MIR_VALIDATED, &mut mir);
    tcx.alloc_steal_mir(mir)
}

fn optimized_mir<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> &'tcx Mir<'tcx> {
    // Borrowck uses `mir_validated`, so we have to force it to
    // execute before we can steal.
    ty::queries::borrowck::force(tcx, DUMMY_SP, def_id);

    let mut mir = tcx.mir_validated(def_id).steal();
    let source = MirSource::from_local_def_id(tcx, def_id);
    transform::run_suite(tcx, source, MIR_OPTIMIZED, &mut mir);
    tcx.alloc_mir(mir)
}
