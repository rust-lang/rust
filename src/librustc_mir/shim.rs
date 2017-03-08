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
use rustc::hir::def_id::DefId;
use rustc::infer;
use rustc::middle::region::ROOT_CODE_EXTENT;
use rustc::mir::*;
use rustc::mir::transform::MirSource;
use rustc::ty::{self, Ty};
use rustc::ty::subst::Subst;
use rustc::ty::maps::Providers;

use rustc_data_structures::indexed_vec::{IndexVec, Idx};

use syntax::abi::Abi;
use syntax::ast;
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
    let did = instance.def_id();
    let span = tcx.def_span(did);
    let param_env =
        tcx.construct_parameter_environment(span, did, ROOT_CODE_EXTENT);

    let result = match instance {
        ty::InstanceDef::Item(..) =>
            bug!("item {:?} passed to make_shim", instance),
        ty::InstanceDef::FnPtrShim(def_id, ty) => {
            let trait_ = tcx.trait_of_item(def_id).unwrap();
            let adjustment = match tcx.lang_items.fn_trait_kind(trait_) {
                Some(ty::ClosureKind::FnOnce) => Adjustment::Identity,
                Some(ty::ClosureKind::FnMut) |
                Some(ty::ClosureKind::Fn) => Adjustment::Deref,
                None => bug!("fn pointer {:?} is not an fn", ty)
            };
            // HACK: we need the "real" argument types for the MIR,
            // but because our substs are (Self, Args), where Args
            // is a tuple, we must include the *concrete* argument
            // types in the MIR. They will be substituted again with
            // the param-substs, but because they are concrete, this
            // will not do any harm.
            let sig = tcx.erase_late_bound_regions(&ty.fn_sig());
            let arg_tys = sig.inputs();

            build_call_shim(
                tcx,
                &param_env,
                def_id,
                adjustment,
                CallKind::Indirect,
                Some(arg_tys)
            )
        }
        ty::InstanceDef::Virtual(def_id, _) => {
            // We are translating a call back to our def-id, which
            // trans::mir knows to turn to an actual virtual call.
            build_call_shim(
                tcx,
                &param_env,
                def_id,
                Adjustment::Identity,
                CallKind::Direct(def_id),
                None
            )
        }
        _ => bug!("unknown shim kind")
    };
    debug!("make_shim({:?}) = {:?}", instance, result);

    let result = tcx.alloc_mir(result);
    // Perma-borrow MIR from shims to prevent mutation.
    mem::forget(result.borrow());
    result
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Adjustment {
    Identity,
    Deref,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum CallKind {
    Indirect,
    Direct(DefId),
}

fn temp_decl(mutability: Mutability, ty: Ty) -> LocalDecl {
    LocalDecl { mutability, ty, name: None, source_info: None }
}

fn local_decls_for_sig<'tcx>(sig: &ty::FnSig<'tcx>)
    -> IndexVec<Local, LocalDecl<'tcx>>
{
    iter::once(temp_decl(Mutability::Mut, sig.output()))
        .chain(sig.inputs().iter().map(
            |ity| temp_decl(Mutability::Not, ity)))
        .collect()
}

/// Build a "call" shim for `def_id`. The shim calls the
/// function specified by `call_kind`, first adjusting its first
/// argument according to `rcvr_adjustment`.
///
/// If `untuple_args` is a vec of types, the second argument of the
/// function will be untupled as these types.
fn build_call_shim<'a, 'tcx>(tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
                             param_env: &ty::ParameterEnvironment<'tcx>,
                             def_id: DefId,
                             rcvr_adjustment: Adjustment,
                             call_kind: CallKind,
                             untuple_args: Option<&[Ty<'tcx>]>)
                             -> Mir<'tcx>
{
    debug!("build_call_shim(def_id={:?}, rcvr_adjustment={:?}, \
            call_kind={:?}, untuple_args={:?})",
           def_id, rcvr_adjustment, call_kind, untuple_args);

    let fn_ty = tcx.item_type(def_id).subst(tcx, param_env.free_substs);
    // Not normalizing here without a param env.
    let sig = tcx.erase_late_bound_regions(&fn_ty.fn_sig());
    let span = tcx.def_span(def_id);

    debug!("build_call_shim: sig={:?}", sig);

    let local_decls = local_decls_for_sig(&sig);
    let source_info = SourceInfo { span, scope: ARGUMENT_VISIBILITY_SCOPE };

    let rcvr_l = Lvalue::Local(Local::new(1+0));

    let return_block_id = BasicBlock::new(1);

    let rcvr = match rcvr_adjustment {
        Adjustment::Identity => Operand::Consume(rcvr_l),
        Adjustment::Deref => Operand::Consume(Lvalue::Projection(
            box Projection { base: rcvr_l, elem: ProjectionElem::Deref }
        ))
    };

    let (callee, mut args) = match call_kind {
        CallKind::Indirect => (rcvr, vec![]),
        CallKind::Direct(def_id) => (
            Operand::Constant(Constant {
                span: span,
                ty: tcx.item_type(def_id).subst(tcx, param_env.free_substs),
                literal: Literal::Item { def_id, substs: param_env.free_substs },
            }),
            vec![rcvr]
        )
    };

    if let Some(untuple_args) = untuple_args {
        args.extend(untuple_args.iter().enumerate().map(|(i, ity)| {
            let arg_lv = Lvalue::Local(Local::new(1+1));
            Operand::Consume(Lvalue::Projection(box Projection {
                base: arg_lv,
                elem: ProjectionElem::Field(Field::new(i), *ity)
            }))
        }));
    } else {
        args.extend((1..sig.inputs().len()).map(|i| {
            Operand::Consume(Lvalue::Local(Local::new(1+i)))
        }));
    }

    let mut blocks = IndexVec::new();
    blocks.push(BasicBlockData {
        statements: vec![],
        terminator: Some(Terminator {
            source_info: source_info,
            kind: TerminatorKind::Call {
                func: callee,
                args: args,
                destination: Some((Lvalue::Local(RETURN_POINTER),
                                   return_block_id)),
                cleanup: None
            }
        }),
        is_cleanup: false
    });
    blocks.push(BasicBlockData {
        statements: vec![],
        terminator: Some(Terminator {
            source_info: source_info,
            kind: TerminatorKind::Return
        }),
        is_cleanup: false
    });

    let mut mir = Mir::new(
        blocks,
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
    if let Abi::RustCall = sig.abi {
        mir.spread_arg = Some(Local::new(sig.inputs().len()));
    }
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
