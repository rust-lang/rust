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
use rustc::middle::const_val::ConstVal;
use rustc::mir::*;
use rustc::mir::transform::MirSource;
use rustc::ty::{self, Ty};
use rustc::ty::subst::{Kind, Subst};
use rustc::ty::maps::Providers;

use rustc_data_structures::indexed_vec::{IndexVec, Idx};

use syntax::abi::Abi;
use syntax::ast;
use syntax_pos::Span;

use std::cell::RefCell;
use std::fmt;
use std::iter;
use std::mem;

use transform::{add_call_guards, no_landing_pads, simplify};
use util::elaborate_drops::{self, DropElaborator, DropStyle, DropFlagMode};
use util::patch::MirPatch;

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

    let mut result = match instance {
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
        ty::InstanceDef::ClosureOnceShim { call_once } => {
            let fn_mut = tcx.lang_items.fn_mut_trait().unwrap();
            let call_mut = tcx.global_tcx()
                .associated_items(fn_mut)
                .find(|it| it.kind == ty::AssociatedKind::Method)
                .unwrap().def_id;

            build_call_shim(
                tcx,
                &param_env,
                call_once,
                Adjustment::RefMut,
                CallKind::Direct(call_mut),
                None
            )
        }
        ty::InstanceDef::DropGlue(def_id, ty) => {
            build_drop_shim(tcx, &param_env, def_id, ty)
        }
        ty::InstanceDef::Intrinsic(_) => {
            bug!("creating shims from intrinsics ({:?}) is unsupported", instance)
        }
    };
        debug!("make_shim({:?}) = untransformed {:?}", instance, result);
        no_landing_pads::no_landing_pads(tcx, &mut result);
        simplify::simplify_cfg(&mut result);
        add_call_guards::add_call_guards(&mut result);
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
    RefMut,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum CallKind {
    Indirect,
    Direct(DefId),
}

fn temp_decl(mutability: Mutability, ty: Ty, span: Span) -> LocalDecl {
    LocalDecl {
        mutability, ty, name: None,
        source_info: SourceInfo { scope: ARGUMENT_VISIBILITY_SCOPE, span },
        is_user_variable: false
    }
}

fn local_decls_for_sig<'tcx>(sig: &ty::FnSig<'tcx>, span: Span)
    -> IndexVec<Local, LocalDecl<'tcx>>
{
    iter::once(temp_decl(Mutability::Mut, sig.output(), span))
        .chain(sig.inputs().iter().map(
            |ity| temp_decl(Mutability::Not, ity, span)))
        .collect()
}

fn build_drop_shim<'a, 'tcx>(tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
                             param_env: &ty::ParameterEnvironment<'tcx>,
                             def_id: DefId,
                             ty: Option<Ty<'tcx>>)
                             -> Mir<'tcx>
{
    debug!("build_drop_shim(def_id={:?}, ty={:?})", def_id, ty);

    let substs = if let Some(ty) = ty {
        tcx.mk_substs(iter::once(Kind::from(ty)))
    } else {
        param_env.free_substs
    };
    let fn_ty = tcx.item_type(def_id).subst(tcx, substs);
    let sig = tcx.erase_late_bound_regions(&fn_ty.fn_sig());
    let span = tcx.def_span(def_id);

    let source_info = SourceInfo { span, scope: ARGUMENT_VISIBILITY_SCOPE };

    let return_block = BasicBlock::new(1);
    let mut blocks = IndexVec::new();
    let block = |blocks: &mut IndexVec<_, _>, kind| {
        blocks.push(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator { source_info, kind }),
            is_cleanup: false
        })
    };
    block(&mut blocks, TerminatorKind::Goto { target: return_block });
    block(&mut blocks, TerminatorKind::Return);

    let mut mir = Mir::new(
        blocks,
        IndexVec::from_elem_n(
            VisibilityScopeData { span: span, parent_scope: None }, 1
        ),
        IndexVec::new(),
        sig.output(),
        local_decls_for_sig(&sig, span),
        sig.inputs().len(),
        vec![],
        span
    );

    if let Some(..) = ty {
        let patch = {
            let mut elaborator = DropShimElaborator {
                mir: &mir,
                patch: MirPatch::new(&mir),
                tcx, param_env
            };
            let dropee = Lvalue::Local(Local::new(1+0)).deref();
            let resume_block = elaborator.patch.resume_block();
            elaborate_drops::elaborate_drop(
                &mut elaborator,
                source_info,
                false,
                &dropee,
                (),
                return_block,
                Some(resume_block),
                START_BLOCK
            );
            elaborator.patch
        };
        patch.apply(&mut mir);
    }

    mir
}

pub struct DropShimElaborator<'a, 'tcx: 'a> {
    mir: &'a Mir<'tcx>,
    patch: MirPatch<'tcx>,
    tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    param_env: &'a ty::ParameterEnvironment<'tcx>,
}

impl<'a, 'tcx> fmt::Debug for DropShimElaborator<'a, 'tcx> {
    fn fmt(&self, _f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        Ok(())
    }
}

impl<'a, 'tcx> DropElaborator<'a, 'tcx> for DropShimElaborator<'a, 'tcx> {
    type Path = ();

    fn patch(&mut self) -> &mut MirPatch<'tcx> { &mut self.patch }
    fn mir(&self) -> &'a Mir<'tcx> { self.mir }
    fn tcx(&self) -> ty::TyCtxt<'a, 'tcx, 'tcx> { self.tcx }
    fn param_env(&self) -> &'a ty::ParameterEnvironment<'tcx> { self.param_env }

    fn drop_style(&self, _path: Self::Path, mode: DropFlagMode) -> DropStyle {
        if let DropFlagMode::Shallow = mode {
            DropStyle::Static
        } else {
            DropStyle::Open
        }
    }

    fn get_drop_flag(&mut self, _path: Self::Path) -> Option<Operand<'tcx>> {
        None
    }

    fn clear_drop_flag(&mut self, _location: Location, _path: Self::Path, _mode: DropFlagMode) {
    }

    fn field_subpath(&self, _path: Self::Path, _field: Field) -> Option<Self::Path> {
        None
    }
    fn deref_subpath(&self, _path: Self::Path) -> Option<Self::Path> {
        None
    }
    fn downcast_subpath(&self, _path: Self::Path, _variant: usize) -> Option<Self::Path> {
        Some(())
    }
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
    let sig = tcx.erase_late_bound_regions(&fn_ty.fn_sig());
    let span = tcx.def_span(def_id);

    debug!("build_call_shim: sig={:?}", sig);

    let mut local_decls = local_decls_for_sig(&sig, span);
    let source_info = SourceInfo { span, scope: ARGUMENT_VISIBILITY_SCOPE };

    let rcvr_arg = Local::new(1+0);
    let rcvr_l = Lvalue::Local(rcvr_arg);
    let mut statements = vec![];

    let rcvr = match rcvr_adjustment {
        Adjustment::Identity => Operand::Consume(rcvr_l),
        Adjustment::Deref => Operand::Consume(rcvr_l.deref()),
        Adjustment::RefMut => {
            // let rcvr = &mut rcvr;
            let ref_rcvr = local_decls.push(temp_decl(
                Mutability::Not,
                tcx.mk_ref(tcx.types.re_erased, ty::TypeAndMut {
                    ty: sig.inputs()[0],
                    mutbl: hir::Mutability::MutMutable
                }),
                span
            ));
            statements.push(Statement {
                source_info: source_info,
                kind: StatementKind::Assign(
                    Lvalue::Local(ref_rcvr),
                    Rvalue::Ref(tcx.types.re_erased, BorrowKind::Mut, rcvr_l)
                )
            });
            Operand::Consume(Lvalue::Local(ref_rcvr))
        }
    };

    let (callee, mut args) = match call_kind {
        CallKind::Indirect => (rcvr, vec![]),
        CallKind::Direct(def_id) => (
            Operand::Constant(Constant {
                span: span,
                ty: tcx.item_type(def_id).subst(tcx, param_env.free_substs),
                literal: Literal::Value {
                    value: ConstVal::Function(def_id, param_env.free_substs),
                },
            }),
            vec![rcvr]
        )
    };

    if let Some(untuple_args) = untuple_args {
        args.extend(untuple_args.iter().enumerate().map(|(i, ity)| {
            let arg_lv = Lvalue::Local(Local::new(1+1));
            Operand::Consume(arg_lv.field(Field::new(i), *ity))
        }));
    } else {
        args.extend((1..sig.inputs().len()).map(|i| {
            Operand::Consume(Lvalue::Local(Local::new(1+i)))
        }));
    }

    let mut blocks = IndexVec::new();
    let block = |blocks: &mut IndexVec<_, _>, statements, kind, is_cleanup| {
        blocks.push(BasicBlockData {
            statements,
            terminator: Some(Terminator { source_info, kind }),
            is_cleanup
        })
    };

    // BB #0
    block(&mut blocks, statements, TerminatorKind::Call {
        func: callee,
        args: args,
        destination: Some((Lvalue::Local(RETURN_POINTER),
                           BasicBlock::new(1))),
        cleanup: if let Adjustment::RefMut = rcvr_adjustment {
            Some(BasicBlock::new(3))
        } else {
            None
        }
    }, false);

    if let Adjustment::RefMut = rcvr_adjustment {
        // BB #1 - drop for Self
        block(&mut blocks, vec![], TerminatorKind::Drop {
            location: Lvalue::Local(rcvr_arg),
            target: BasicBlock::new(2),
            unwind: None
        }, false);
    }
    // BB #1/#2 - return
    block(&mut blocks, vec![], TerminatorKind::Return, false);
    if let Adjustment::RefMut = rcvr_adjustment {
        // BB #3 - drop if closure panics
        block(&mut blocks, vec![], TerminatorKind::Drop {
            location: Lvalue::Local(rcvr_arg),
            target: BasicBlock::new(4),
            unwind: None
        }, true);

        // BB #4 - resume
        block(&mut blocks, vec![], TerminatorKind::Resume, true);
    }

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

    let local_decls = local_decls_for_sig(&sig, span);

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
