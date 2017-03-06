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
use rustc::ty::{self, Ty};
use rustc::ty::maps::Providers;

use rustc_data_structures::indexed_vec::{IndexVec, Idx};

use syntax::abi::Abi;
use syntax::ast;
use syntax::codemap::DUMMY_SP;
use syntax_pos::Span;

use std::cell::RefCell;
use std::iter;
use std::mem;

pub fn provide(providers: &mut Providers) {
    providers.mir_shims = make_shim;
}

fn make_shim<'a, 'tcx>(tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
                       instance: ty::InstanceDef<'tcx>)
                       -> &'tcx RefCell<Mir<'tcx>>
{
    debug!("make_shim({:?})", instance);
    let result = match instance {
        ty::InstanceDef::Item(..) =>
            bug!("item {:?} passed to make_shim", instance),
        ty::InstanceDef::FnPtrShim(_, ty) => {
            build_fn_ptr_shim(tcx, ty, instance.def_ty(tcx))
        }
    };
    debug!("make_shim({:?}) = {:?}", instance, result);

    let result = tcx.alloc_mir(result);
    // Perma-borrow MIR from shims to prevent mutation.
    mem::forget(result.borrow());
    result
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


fn build_fn_ptr_shim<'a, 'tcx>(tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
                               fn_ty: Ty<'tcx>,
                               sig_ty: Ty<'tcx>)
                               -> Mir<'tcx>
{
    debug!("build_fn_ptr_shim(fn_ty={:?}, sig_ty={:?})", fn_ty, sig_ty);
    let trait_sig = match sig_ty.sty {
        ty::TyFnDef(_, _, fty) => tcx.erase_late_bound_regions(&fty),
        _ => bug!("unexpected type for shim {:?}", sig_ty)
    };

    let self_ty = match trait_sig.inputs()[0].sty {
        ty::TyParam(..) => fn_ty,
        ty::TyRef(r, mt) => tcx.mk_ref(r, ty::TypeAndMut {
            ty: fn_ty,
            mutbl: mt.mutbl
        }),
        _ => bug!("unexpected self_ty {:?}", trait_sig),
    };

    let fn_ptr_sig = match fn_ty.sty {
        ty::TyFnPtr(fty) |
        ty::TyFnDef(_, _, fty) =>
            tcx.erase_late_bound_regions_and_normalize(&fty),
        _ => bug!("non-fn-ptr {:?} in build_fn_ptr_shim", fn_ty)
    };

    let sig = tcx.mk_fn_sig(
        [
            self_ty,
            tcx.intern_tup(fn_ptr_sig.inputs(), false)
        ].iter().cloned(),
        fn_ptr_sig.output(),
        false,
        hir::Unsafety::Normal,
        Abi::RustCall,
    );

    let local_decls = local_decls_for_sig(&sig);
    let source_info = SourceInfo {
        span: DUMMY_SP,
        scope: ARGUMENT_VISIBILITY_SCOPE
    };

    let fn_ptr = Lvalue::Local(Local::new(1+0));
    let fn_ptr = match trait_sig.inputs()[0].sty {
        ty::TyParam(..) => fn_ptr,
        ty::TyRef(..) => Lvalue::Projection(box Projection {
            base: fn_ptr, elem: ProjectionElem::Deref
        }),
        _ => bug!("unexpected self_ty {:?}", trait_sig),
    };
    let fn_args = Local::new(1+1);

    let return_block_id = BasicBlock::new(1);

    // return = ADT(arg0, arg1, ...); return
    let start_block = BasicBlockData {
        statements: vec![],
        terminator: Some(Terminator {
            source_info: source_info,
            kind: TerminatorKind::Call {
                func: Operand::Consume(fn_ptr),
                args: fn_ptr_sig.inputs().iter().enumerate().map(|(i, ity)| {
                    Operand::Consume(Lvalue::Projection(box Projection {
                        base: Lvalue::Local(fn_args),
                        elem: ProjectionElem::Field(
                            Field::new(i), *ity
                        )
                    }))
                }).collect(),
                // FIXME: can we pass a Some destination for an uninhabited ty?
                destination: Some((Lvalue::Local(RETURN_POINTER),
                                   return_block_id)),
                cleanup: None
            }
        }),
        is_cleanup: false
    };
    let return_block = BasicBlockData {
        statements: vec![],
        terminator: Some(Terminator {
            source_info: source_info,
            kind: TerminatorKind::Return
        }),
        is_cleanup: false
    };

    let mut mir = Mir::new(
        vec![start_block, return_block].into_iter().collect(),
        IndexVec::from_elem_n(
            VisibilityScopeData { span: DUMMY_SP, parent_scope: None }, 1
        ),
        IndexVec::new(),
        sig.output(),
        local_decls,
        sig.inputs().len(),
        vec![],
        DUMMY_SP
    );
    mir.spread_arg = Some(fn_args);
    mir
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
