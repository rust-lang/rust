// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir;
use rustc::infer;
use rustc::mir::*;
use rustc::mir::transform::MirSource;
use rustc::ty;
use rustc::ty::maps::Providers;

use rustc_data_structures::indexed_vec::{IndexVec, Idx};

use syntax::ast;
use syntax_pos::Span;

use std::cell::RefCell;
use std::iter;

pub fn provide(providers: &mut Providers) {
    providers.mir_shims = make_shim;
}

fn make_shim<'a, 'tcx>(_tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
                       instance: ty::InstanceDef<'tcx>)
                       -> &'tcx RefCell<Mir<'tcx>>
{
    match instance {
        ty::InstanceDef::Item(..) =>
            bug!("item {:?} passed to make_shim", instance),
        ty::InstanceDef::FnPtrShim(..) => unimplemented!()
    }
}

fn local_decls_for_sig<'tcx>(sig: &ty::FnSig<'tcx>)
    -> IndexVec<Local, LocalDecl<'tcx>>
{
    iter::once(LocalDecl {
        mutability: Mutability::Mut,
        ty: sig.output(),
        name: None,
        source_info: None
    }).chain(sig.inputs().iter().map(|ity| LocalDecl {
        mutability: Mutability::Not,
        ty: *ity,
        name: None,
        source_info: None,
    })).collect()
}

pub fn build_adt_ctor<'a, 'gcx, 'tcx>(infcx: &infer::InferCtxt<'a, 'gcx, 'tcx>,
                                      ctor_id: ast::NodeId,
                                      fields: &[hir::StructField],
                                      span: Span)
                                      -> (Mir<'tcx>, MirSource)
{
    let tcx = infcx.tcx;
    let def_id = tcx.hir.local_def_id(ctor_id);
    let sig = match tcx.item_type(def_id).sty {
        ty::TyFnDef(_, _, fty) => tcx.no_late_bound_regions(&fty)
            .expect("LBR in ADT constructor signature"),
        _ => bug!("unexpected type for ctor {:?}", def_id)
    };
    let sig = tcx.erase_regions(&sig);

    let (adt_def, substs) = match sig.output().sty {
        ty::TyAdt(adt_def, substs) => (adt_def, substs),
        _ => bug!("unexpected type for ADT ctor {:?}", sig.output())
    };

    debug!("build_ctor: def_id={:?} sig={:?} fields={:?}", def_id, sig, fields);

    let local_decls = local_decls_for_sig(&sig);

    let source_info = SourceInfo {
        span: span,
        scope: ARGUMENT_VISIBILITY_SCOPE
    };

    let variant_no = if adt_def.is_enum() {
        adt_def.variant_index_with_id(def_id)
    } else {
        0
    };

    // return = ADT(arg0, arg1, ...); return
    let start_block = BasicBlockData {
        statements: vec![Statement {
            source_info: source_info,
            kind: StatementKind::Assign(
                Lvalue::Local(RETURN_POINTER),
                Rvalue::Aggregate(
                    AggregateKind::Adt(adt_def, variant_no, substs, None),
                    (1..sig.inputs().len()+1).map(|i| {
                        Operand::Consume(Lvalue::Local(Local::new(i)))
                    }).collect()
                )
            )
        }],
        terminator: Some(Terminator {
            source_info: source_info,
            kind: TerminatorKind::Return,
        }),
        is_cleanup: false
    };

    let mir = Mir::new(
        IndexVec::from_elem_n(start_block, 1),
        IndexVec::from_elem_n(
            VisibilityScopeData { span: span, parent_scope: None }, 1
        ),
        IndexVec::new(),
        sig.output(),
        local_decls,
        sig.inputs().len(),
        vec![],
        span
    );
    (mir, MirSource::Fn(ctor_id))
}
