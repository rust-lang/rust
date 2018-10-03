// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![allow(non_camel_case_types, non_snake_case)]

use rustc::ty::{self, Ty, TyCtxt};
use syntax_pos::{DUMMY_SP, Span};

use rustc::hir::def_id::DefId;
use rustc::middle::lang_items::LangItem;
use base;
use interfaces::*;

use rustc::hir;
use interfaces::BuilderMethods;

use std::iter;

use rustc_target::spec::abi::Abi;


pub fn type_needs_drop<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> bool {
    ty.needs_drop(tcx, ty::ParamEnv::reveal_all())
}

pub fn type_is_sized<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> bool {
    ty.is_sized(tcx.at(DUMMY_SP), ty::ParamEnv::reveal_all())
}

pub fn type_is_freeze<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> bool {
    ty.is_freeze(tcx, ty::ParamEnv::reveal_all(), DUMMY_SP)
}

pub struct OperandBundleDef<'a, V> {
    pub name: &'a str,
    pub val: V
}

impl<'a, V> OperandBundleDef<'a, V> {
    pub fn new(name: &'a str, val: V) -> Self {
        OperandBundleDef {
            name,
            val
        }
    }
}



/// A structure representing an active landing pad for the duration of a basic
/// block.
///
/// Each `Block` may contain an instance of this, indicating whether the block
/// is part of a landing pad or not. This is used to make decision about whether
/// to emit `invoke` instructions (e.g. in a landing pad we don't continue to
/// use `invoke`) and also about various function call metadata.
///
/// For GNU exceptions (`landingpad` + `resume` instructions) this structure is
/// just a bunch of `None` instances (not too interesting), but for MSVC
/// exceptions (`cleanuppad` + `cleanupret` instructions) this contains data.
/// When inside of a landing pad, each function call in LLVM IR needs to be
/// annotated with which landing pad it's a part of. This is accomplished via
/// the `OperandBundleDef` value created for MSVC landing pads.
pub struct Funclet<'ll, V> {
    cleanuppad: V,
    operand: OperandBundleDef<'ll, V>,
}

impl<'ll, V : CodegenObject> Funclet<'ll, V> {
    pub fn new(cleanuppad: V) -> Self {
        Funclet {
            cleanuppad,
            operand: OperandBundleDef::new("funclet", cleanuppad),
        }
    }

    pub fn cleanuppad(&self) -> V {
        self.cleanuppad
    }

    pub fn bundle(&self) -> &OperandBundleDef<'ll, V> {
        &self.operand
    }
}

pub enum IntPredicate {
    IntEQ,
    IntNE,
    IntUGT,
    IntUGE,
    IntULT,
    IntULE,
    IntSGT,
    IntSGE,
    IntSLT,
    IntSLE
}


#[allow(dead_code)]
pub enum RealPredicate {
    RealPredicateFalse,
    RealOEQ,
    RealOGT,
    RealOGE,
    RealOLT,
    RealOLE,
    RealONE,
    RealORD,
    RealUNO,
    RealUEQ,
    RealUGT,
    RealUGE,
    RealULT,
    RealULE,
    RealUNE,
    RealPredicateTrue
}

pub enum AtomicRmwBinOp {
    AtomicXchg,
    AtomicAdd,
    AtomicSub,
    AtomicAnd,
    AtomicNand,
    AtomicOr,
    AtomicXor,
    AtomicMax,
    AtomicMin,
    AtomicUMax,
    AtomicUMin
}

pub enum AtomicOrdering {
    #[allow(dead_code)]
    NotAtomic,
    Unordered,
    Monotonic,
    // Consume,  // Not specified yet.
    Acquire,
    Release,
    AcquireRelease,
    SequentiallyConsistent,
}

pub enum SynchronizationScope {
    // FIXME: figure out if this variant is needed at all.
    #[allow(dead_code)]
    Other,
    SingleThread,
    CrossThread,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum TypeKind {
    Void,
    Half,
    Float,
    Double,
    X86_FP80,
    FP128,
    PPC_FP128,
    Label,
    Integer,
    Function,
    Struct,
    Array,
    Pointer,
    Vector,
    Metadata,
    X86_MMX,
    Token,
}

// FIXME(mw): Anything that is produced via DepGraph::with_task() must implement
//            the HashStable trait. Normally DepGraph::with_task() calls are
//            hidden behind queries, but CGU creation is a special case in two
//            ways: (1) it's not a query and (2) CGU are output nodes, so their
//            Fingerprints are not actually needed. It remains to be clarified
//            how exactly this case will be handled in the red/green system but
//            for now we content ourselves with providing a no-op HashStable
//            implementation for CGUs.
mod temp_stable_hash_impls {
    use rustc_data_structures::stable_hasher::{StableHasherResult, StableHasher,
                                               HashStable};
    use ModuleCodegen;

    impl<HCX, M> HashStable<HCX> for ModuleCodegen<M> {
        fn hash_stable<W: StableHasherResult>(&self,
                                              _: &mut HCX,
                                              _: &mut StableHasher<W>) {
            // do nothing
        }
    }
}



// To avoid UB from LLVM, these two functions mask RHS with an
// appropriate mask unconditionally (i.e. the fallback behavior for
// all shifts). For 32- and 64-bit types, this matches the semantics
// of Java. (See related discussion on #1877 and #10183.)

pub fn build_unchecked_lshift<'a, 'll: 'a, 'tcx: 'll, Bx: BuilderMethods<'a, 'll, 'tcx>>(
    bx: &Bx,
    lhs: <Bx::CodegenCx as Backend<'ll>>::Value,
    rhs: <Bx::CodegenCx as Backend<'ll>>::Value
) -> <Bx::CodegenCx as Backend<'ll>>::Value {
    let rhs = base::cast_shift_expr_rhs(bx, hir::BinOpKind::Shl, lhs, rhs);
    // #1877, #10183: Ensure that input is always valid
    let rhs = shift_mask_rhs(bx, rhs);
    bx.shl(lhs, rhs)
}

pub fn build_unchecked_rshift<'a, 'll: 'a, 'tcx: 'll, Bx: BuilderMethods<'a, 'll, 'tcx>>(
    bx: &Bx,
    lhs_t: Ty<'tcx>,
    lhs: <Bx::CodegenCx as Backend<'ll>>::Value,
    rhs: <Bx::CodegenCx as Backend<'ll>>::Value
) -> <Bx::CodegenCx as Backend<'ll>>::Value {
    let rhs = base::cast_shift_expr_rhs(bx, hir::BinOpKind::Shr, lhs, rhs);
    // #1877, #10183: Ensure that input is always valid
    let rhs = shift_mask_rhs(bx, rhs);
    let is_signed = lhs_t.is_signed();
    if is_signed {
        bx.ashr(lhs, rhs)
    } else {
        bx.lshr(lhs, rhs)
    }
}

fn shift_mask_rhs<'a, 'll: 'a, 'tcx: 'll, Bx: BuilderMethods<'a, 'll, 'tcx>>(
    bx: &Bx,
    rhs: <Bx::CodegenCx as Backend<'ll>>::Value
) -> <Bx::CodegenCx as Backend<'ll>>::Value {
    let rhs_llty = bx.cx().val_ty(rhs);
    bx.and(rhs, shift_mask_val(bx, rhs_llty, rhs_llty, false))
}

pub fn shift_mask_val<'a, 'll: 'a, 'tcx: 'll, Bx: BuilderMethods<'a, 'll, 'tcx>>(
    bx: &Bx,
    llty: <Bx::CodegenCx as Backend<'ll>>::Type,
    mask_llty: <Bx::CodegenCx as Backend<'ll>>::Type,
    invert: bool
) -> <Bx::CodegenCx as Backend<'ll>>::Value {
    let kind = bx.cx().type_kind(llty);
    match kind {
        TypeKind::Integer => {
            // i8/u8 can shift by at most 7, i16/u16 by at most 15, etc.
            let val = bx.cx().int_width(llty) - 1;
            if invert {
                bx.cx().const_int(mask_llty, !val as i64)
            } else {
                bx.cx().const_uint(mask_llty, val)
            }
        },
        TypeKind::Vector => {
            let mask = shift_mask_val(
                bx,
                bx.cx().element_type(llty),
                bx.cx().element_type(mask_llty),
                invert
            );
            bx.vector_splat(bx.cx().vector_length(mask_llty), mask)
        },
        _ => bug!("shift_mask_val: expected Integer or Vector, found {:?}", kind),
    }
}

pub fn ty_fn_sig<'ll, 'tcx:'ll, Cx: CodegenMethods<'ll, 'tcx>>(
    cx: &Cx,
    ty: Ty<'tcx>
) -> ty::PolyFnSig<'tcx> {
    match ty.sty {
        ty::FnDef(..) |
        // Shims currently have type FnPtr. Not sure this should remain.
        ty::FnPtr(_) => ty.fn_sig(*cx.tcx()),
        ty::Closure(def_id, substs) => {
            let tcx = *cx.tcx();
            let sig = substs.closure_sig(def_id, tcx);

            let env_ty = tcx.closure_env_ty(def_id, substs).unwrap();
            sig.map_bound(|sig| tcx.mk_fn_sig(
                iter::once(*env_ty.skip_binder()).chain(sig.inputs().iter().cloned()),
                sig.output(),
                sig.variadic,
                sig.unsafety,
                sig.abi
            ))
        }
        ty::Generator(def_id, substs, _) => {
            let tcx = *cx.tcx();
            let sig = substs.poly_sig(def_id, tcx);

            let env_region = ty::ReLateBound(ty::INNERMOST, ty::BrEnv);
            let env_ty = tcx.mk_mut_ref(tcx.mk_region(env_region), ty);

            sig.map_bound(|sig| {
                let state_did = tcx.lang_items().gen_state().unwrap();
                let state_adt_ref = tcx.adt_def(state_did);
                let state_substs = tcx.intern_substs(&[
                    sig.yield_ty.into(),
                    sig.return_ty.into(),
                ]);
                let ret_ty = tcx.mk_adt(state_adt_ref, state_substs);

                tcx.mk_fn_sig(iter::once(env_ty),
                    ret_ty,
                    false,
                    hir::Unsafety::Normal,
                    Abi::Rust
                )
            })
        }
        _ => bug!("unexpected type {:?} to ty_fn_sig", ty)
    }
}

pub fn langcall(tcx: TyCtxt,
                span: Option<Span>,
                msg: &str,
                li: LangItem)
                -> DefId {
    match tcx.lang_items().require(li) {
        Ok(id) => id,
        Err(s) => {
            let msg = format!("{} {}", msg, s);
            match span {
                Some(span) => tcx.sess.span_fatal(span, &msg[..]),
                None => tcx.sess.fatal(&msg[..]),
            }
        }
    }
}
