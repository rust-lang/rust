// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{self, ValueRef};
use rustc::middle::const_val::ConstVal;
use rustc_const_eval::{ErrKind, ConstEvalErr, report_const_eval_err};
use rustc_const_math::ConstInt::*;
use rustc_const_math::ConstFloat::*;
use rustc_const_math::{ConstInt, ConstMathErr};
use rustc::hir::def_id::DefId;
use rustc::infer::TransNormalize;
use rustc::mir;
use rustc::mir::tcx::LvalueTy;
use rustc::ty::{self, layout, Ty, TyCtxt, TypeFoldable};
use rustc::ty::cast::{CastTy, IntTy};
use rustc::ty::subst::Substs;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use {abi, adt, base, Disr, machine};
use callee::Callee;
use builder::Builder;
use common::{self, CrateContext, const_get_elt, val_ty};
use common::{C_array, C_bool, C_bytes, C_floating_f64, C_integral, C_big_integral};
use common::{C_null, C_struct, C_str_slice, C_undef, C_uint, C_vector, is_undef};
use common::const_to_opt_u128;
use consts;
use monomorphize::{self, Instance};
use type_of;
use type_::Type;
use value::Value;

use syntax_pos::Span;

use std::fmt;
use std::ptr;

use super::operand::{OperandRef, OperandValue};
use super::MirContext;

use rustc_i128::{u128, i128};

/// A sized constant rvalue.
/// The LLVM type might not be the same for a single Rust type,
/// e.g. each enum variant would have its own LLVM struct type.
#[derive(Copy, Clone)]
pub struct Const<'tcx> {
    pub llval: ValueRef,
    pub ty: Ty<'tcx>
}

impl<'tcx> Const<'tcx> {
    pub fn new(llval: ValueRef, ty: Ty<'tcx>) -> Const<'tcx> {
        Const {
            llval: llval,
            ty: ty
        }
    }

    /// Translate ConstVal into a LLVM constant value.
    pub fn from_constval<'a>(ccx: &CrateContext<'a, 'tcx>,
                             cv: ConstVal,
                             ty: Ty<'tcx>)
                             -> Const<'tcx> {
        let llty = type_of::type_of(ccx, ty);
        let val = match cv {
            ConstVal::Float(F32(v)) => C_floating_f64(v as f64, llty),
            ConstVal::Float(F64(v)) => C_floating_f64(v, llty),
            ConstVal::Float(FInfer {..}) => bug!("MIR must not use `{:?}`", cv),
            ConstVal::Bool(v) => C_bool(ccx, v),
            ConstVal::Integral(I8(v)) => C_integral(Type::i8(ccx), v as u64, true),
            ConstVal::Integral(I16(v)) => C_integral(Type::i16(ccx), v as u64, true),
            ConstVal::Integral(I32(v)) => C_integral(Type::i32(ccx), v as u64, true),
            ConstVal::Integral(I64(v)) => C_integral(Type::i64(ccx), v as u64, true),
            ConstVal::Integral(I128(v)) => C_big_integral(Type::i128(ccx), v as u128, true),
            ConstVal::Integral(Isize(v)) => {
                let i = v.as_i64(ccx.tcx().sess.target.int_type);
                C_integral(Type::int(ccx), i as u64, true)
            },
            ConstVal::Integral(U8(v)) => C_integral(Type::i8(ccx), v as u64, false),
            ConstVal::Integral(U16(v)) => C_integral(Type::i16(ccx), v as u64, false),
            ConstVal::Integral(U32(v)) => C_integral(Type::i32(ccx), v as u64, false),
            ConstVal::Integral(U64(v)) => C_integral(Type::i64(ccx), v, false),
            ConstVal::Integral(U128(v)) => C_big_integral(Type::i128(ccx), v, false),
            ConstVal::Integral(Usize(v)) => {
                let u = v.as_u64(ccx.tcx().sess.target.uint_type);
                C_integral(Type::int(ccx), u, false)
            },
            ConstVal::Integral(Infer(_)) |
            ConstVal::Integral(InferSigned(_)) => bug!("MIR must not use `{:?}`", cv),
            ConstVal::Str(ref v) => C_str_slice(ccx, v.clone()),
            ConstVal::ByteStr(ref v) => consts::addr_of(ccx, C_bytes(ccx, v), 1, "byte_str"),
            ConstVal::Struct(_) | ConstVal::Tuple(_) |
            ConstVal::Array(..) | ConstVal::Repeat(..) |
            ConstVal::Function(_) => {
                bug!("MIR must not use `{:?}` (which refers to a local ID)", cv)
            }
            ConstVal::Char(c) => C_integral(Type::char(ccx), c as u64, false),
        };

        assert!(!ty.has_erasable_regions());

        Const::new(val, ty)
    }

    fn get_pair(&self) -> (ValueRef, ValueRef) {
        (const_get_elt(self.llval, &[0]),
         const_get_elt(self.llval, &[1]))
    }

    fn get_fat_ptr(&self) -> (ValueRef, ValueRef) {
        assert_eq!(abi::FAT_PTR_ADDR, 0);
        assert_eq!(abi::FAT_PTR_EXTRA, 1);
        self.get_pair()
    }

    fn as_lvalue(&self) -> ConstLvalue<'tcx> {
        ConstLvalue {
            base: Base::Value(self.llval),
            llextra: ptr::null_mut(),
            ty: self.ty
        }
    }

    pub fn to_operand<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> OperandRef<'tcx> {
        let llty = type_of::immediate_type_of(ccx, self.ty);
        let llvalty = val_ty(self.llval);

        let val = if llty == llvalty && common::type_is_imm_pair(ccx, self.ty) {
            let (a, b) = self.get_pair();
            OperandValue::Pair(a, b)
        } else if llty == llvalty && common::type_is_immediate(ccx, self.ty) {
            // If the types match, we can use the value directly.
            OperandValue::Immediate(self.llval)
        } else {
            // Otherwise, or if the value is not immediate, we create
            // a constant LLVM global and cast its address if necessary.
            let align = type_of::align_of(ccx, self.ty);
            let ptr = consts::addr_of(ccx, self.llval, align, "const");
            OperandValue::Ref(consts::ptrcast(ptr, llty.ptr_to()))
        };

        OperandRef {
            val: val,
            ty: self.ty
        }
    }
}

impl<'tcx> fmt::Debug for Const<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Const({:?}: {:?})", Value(self.llval), self.ty)
    }
}

#[derive(Copy, Clone)]
enum Base {
    /// A constant value without an unique address.
    Value(ValueRef),

    /// String literal base pointer (cast from array).
    Str(ValueRef),

    /// The address of a static.
    Static(ValueRef)
}

/// An lvalue as seen from a constant.
#[derive(Copy, Clone)]
struct ConstLvalue<'tcx> {
    base: Base,
    llextra: ValueRef,
    ty: Ty<'tcx>
}

impl<'tcx> ConstLvalue<'tcx> {
    fn to_const(&self, span: Span) -> Const<'tcx> {
        match self.base {
            Base::Value(val) => Const::new(val, self.ty),
            Base::Str(ptr) => {
                span_bug!(span, "loading from `str` ({:?}) in constant",
                          Value(ptr))
            }
            Base::Static(val) => {
                span_bug!(span, "loading from `static` ({:?}) in constant",
                          Value(val))
            }
        }
    }

    pub fn len<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> ValueRef {
        match self.ty.sty {
            ty::TyArray(_, n) => C_uint(ccx, n),
            ty::TySlice(_) | ty::TyStr => {
                assert!(self.llextra != ptr::null_mut());
                self.llextra
            }
            _ => bug!("unexpected type `{}` in ConstLvalue::len", self.ty)
        }
    }
}

/// Machinery for translating a constant's MIR to LLVM values.
/// FIXME(eddyb) use miri and lower its allocations to LLVM.
struct MirConstContext<'a, 'tcx: 'a> {
    ccx: &'a CrateContext<'a, 'tcx>,
    mir: &'a mir::Mir<'tcx>,

    /// Type parameters for const fn and associated constants.
    substs: &'tcx Substs<'tcx>,

    /// Values of locals in a constant or const fn.
    locals: IndexVec<mir::Local, Option<Const<'tcx>>>
}


impl<'a, 'tcx> MirConstContext<'a, 'tcx> {
    fn new(ccx: &'a CrateContext<'a, 'tcx>,
           mir: &'a mir::Mir<'tcx>,
           substs: &'tcx Substs<'tcx>,
           args: IndexVec<mir::Local, Const<'tcx>>)
           -> MirConstContext<'a, 'tcx> {
        let mut context = MirConstContext {
            ccx: ccx,
            mir: mir,
            substs: substs,
            locals: (0..mir.local_decls.len()).map(|_| None).collect(),
        };
        for (i, arg) in args.into_iter().enumerate() {
            // Locals after local 0 are the function arguments
            let index = mir::Local::new(i + 1);
            context.locals[index] = Some(arg);
        }
        context
    }

    fn trans_def(ccx: &'a CrateContext<'a, 'tcx>,
                 instance: Instance<'tcx>,
                 args: IndexVec<mir::Local, Const<'tcx>>)
                 -> Result<Const<'tcx>, ConstEvalErr> {
        let instance = instance.resolve_const(ccx.shared());
        let mir = ccx.tcx().item_mir(instance.def);
        MirConstContext::new(ccx, &mir, instance.substs, args).trans()
    }

    fn monomorphize<T>(&self, value: &T) -> T
        where T: TransNormalize<'tcx>
    {
        monomorphize::apply_param_substs(self.ccx.shared(),
                                         self.substs,
                                         value)
    }

    fn trans(&mut self) -> Result<Const<'tcx>, ConstEvalErr> {
        let tcx = self.ccx.tcx();
        let mut bb = mir::START_BLOCK;

        // Make sure to evaluate all statemenets to
        // report as many errors as we possibly can.
        let mut failure = Ok(());

        loop {
            let data = &self.mir[bb];
            for statement in &data.statements {
                let span = statement.source_info.span;
                match statement.kind {
                    mir::StatementKind::Assign(ref dest, ref rvalue) => {
                        let ty = dest.ty(self.mir, tcx);
                        let ty = self.monomorphize(&ty).to_ty(tcx);
                        match self.const_rvalue(rvalue, ty, span) {
                            Ok(value) => self.store(dest, value, span),
                            Err(err) => if failure.is_ok() { failure = Err(err); }
                        }
                    }
                    mir::StatementKind::StorageLive(_) |
                    mir::StatementKind::StorageDead(_) |
                    mir::StatementKind::Nop => {}
                    mir::StatementKind::SetDiscriminant{ .. } => {
                        span_bug!(span, "SetDiscriminant should not appear in constants?");
                    }
                }
            }

            let terminator = data.terminator();
            let span = terminator.source_info.span;
            bb = match terminator.kind {
                mir::TerminatorKind::Drop { target, .. } | // No dropping.
                mir::TerminatorKind::Goto { target } => target,
                mir::TerminatorKind::Return => {
                    failure?;
                    return Ok(self.locals[mir::RETURN_POINTER].unwrap_or_else(|| {
                        span_bug!(span, "no returned value in constant");
                    }));
                }

                mir::TerminatorKind::Assert { ref cond, expected, ref msg, target, .. } => {
                    let cond = self.const_operand(cond, span)?;
                    let cond_bool = common::const_to_uint(cond.llval) != 0;
                    if cond_bool != expected {
                        let err = match *msg {
                            mir::AssertMessage::BoundsCheck { ref len, ref index } => {
                                let len = self.const_operand(len, span)?;
                                let index = self.const_operand(index, span)?;
                                ErrKind::IndexOutOfBounds {
                                    len: common::const_to_uint(len.llval),
                                    index: common::const_to_uint(index.llval)
                                }
                            }
                            mir::AssertMessage::Math(ref err) => {
                                ErrKind::Math(err.clone())
                            }
                        };

                        let err = ConstEvalErr{ span: span, kind: err };
                        report_const_eval_err(tcx, &err, span, "expression").emit();
                        failure = Err(err);
                    }
                    target
                }

                mir::TerminatorKind::Call { ref func, ref args, ref destination, .. } => {
                    let fn_ty = func.ty(self.mir, tcx);
                    let fn_ty = self.monomorphize(&fn_ty);
                    let instance = match fn_ty.sty {
                        ty::TyFnDef(def_id, substs, _) => {
                            Instance::new(def_id, substs)
                        }
                        _ => span_bug!(span, "calling {:?} (of type {}) in constant",
                                       func, fn_ty)
                    };

                    let mut const_args = IndexVec::with_capacity(args.len());
                    for arg in args {
                        match self.const_operand(arg, span) {
                            Ok(arg) => { const_args.push(arg); },
                            Err(err) => if failure.is_ok() { failure = Err(err); }
                        }
                    }
                    if let Some((ref dest, target)) = *destination {
                        match MirConstContext::trans_def(self.ccx, instance, const_args) {
                            Ok(value) => self.store(dest, value, span),
                            Err(err) => if failure.is_ok() { failure = Err(err); }
                        }
                        target
                    } else {
                        span_bug!(span, "diverging {:?} in constant", terminator.kind);
                    }
                }
                _ => span_bug!(span, "{:?} in constant", terminator.kind)
            };
        }
    }

    fn store(&mut self, dest: &mir::Lvalue<'tcx>, value: Const<'tcx>, span: Span) {
        if let mir::Lvalue::Local(index) = *dest {
            self.locals[index] = Some(value);
        } else {
            span_bug!(span, "assignment to {:?} in constant", dest);
        }
    }

    fn const_lvalue(&self, lvalue: &mir::Lvalue<'tcx>, span: Span)
                    -> Result<ConstLvalue<'tcx>, ConstEvalErr> {
        let tcx = self.ccx.tcx();

        if let mir::Lvalue::Local(index) = *lvalue {
            return Ok(self.locals[index].unwrap_or_else(|| {
                span_bug!(span, "{:?} not initialized", lvalue)
            }).as_lvalue());
        }

        let lvalue = match *lvalue {
            mir::Lvalue::Local(_)  => bug!(), // handled above
            mir::Lvalue::Static(def_id) => {
                ConstLvalue {
                    base: Base::Static(consts::get_static(self.ccx, def_id)),
                    llextra: ptr::null_mut(),
                    ty: lvalue.ty(self.mir, tcx).to_ty(tcx)
                }
            }
            mir::Lvalue::Projection(ref projection) => {
                let tr_base = self.const_lvalue(&projection.base, span)?;
                let projected_ty = LvalueTy::Ty { ty: tr_base.ty }
                    .projection_ty(tcx, &projection.elem);
                let base = tr_base.to_const(span);
                let projected_ty = self.monomorphize(&projected_ty).to_ty(tcx);
                let is_sized = self.ccx.shared().type_is_sized(projected_ty);

                let (projected, llextra) = match projection.elem {
                    mir::ProjectionElem::Deref => {
                        let (base, extra) = if is_sized {
                            (base.llval, ptr::null_mut())
                        } else {
                            base.get_fat_ptr()
                        };
                        if self.ccx.statics().borrow().contains_key(&base) {
                            (Base::Static(base), extra)
                        } else if let ty::TyStr = projected_ty.sty {
                            (Base::Str(base), extra)
                        } else {
                            let v = base;
                            let v = self.ccx.const_unsized().borrow().get(&v).map_or(v, |&v| v);
                            let mut val = unsafe { llvm::LLVMGetInitializer(v) };
                            if val.is_null() {
                                span_bug!(span, "dereference of non-constant pointer `{:?}`",
                                          Value(base));
                            }
                            if projected_ty.is_bool() {
                                unsafe {
                                    val = llvm::LLVMConstTrunc(val, Type::i1(self.ccx).to_ref());
                                }
                            }
                            (Base::Value(val), extra)
                        }
                    }
                    mir::ProjectionElem::Field(ref field, _) => {
                        let llprojected = adt::const_get_field(self.ccx, tr_base.ty, base.llval,
                                                               Disr(0), field.index());
                        let llextra = if is_sized {
                            ptr::null_mut()
                        } else {
                            tr_base.llextra
                        };
                        (Base::Value(llprojected), llextra)
                    }
                    mir::ProjectionElem::Index(ref index) => {
                        let llindex = self.const_operand(index, span)?.llval;

                        let iv = if let Some(iv) = common::const_to_opt_u128(llindex, false) {
                            iv
                        } else {
                            span_bug!(span, "index is not an integer-constant expression")
                        };

                        // Produce an undef instead of a LLVM assertion on OOB.
                        let len = common::const_to_uint(tr_base.len(self.ccx));
                        let llelem = if iv < len as u128 {
                            const_get_elt(base.llval, &[iv as u32])
                        } else {
                            C_undef(type_of::type_of(self.ccx, projected_ty))
                        };

                        (Base::Value(llelem), ptr::null_mut())
                    }
                    _ => span_bug!(span, "{:?} in constant", projection.elem)
                };
                ConstLvalue {
                    base: projected,
                    llextra: llextra,
                    ty: projected_ty
                }
            }
        };
        Ok(lvalue)
    }

    fn const_operand(&self, operand: &mir::Operand<'tcx>, span: Span)
                     -> Result<Const<'tcx>, ConstEvalErr> {
        debug!("const_operand({:?} @ {:?})", operand, span);
        let result = match *operand {
            mir::Operand::Consume(ref lvalue) => {
                Ok(self.const_lvalue(lvalue, span)?.to_const(span))
            }

            mir::Operand::Constant(ref constant) => {
                let ty = self.monomorphize(&constant.ty);
                match constant.literal.clone() {
                    mir::Literal::Item { def_id, substs } => {
                        // Shortcut for zero-sized types, including function item
                        // types, which would not work with MirConstContext.
                        if common::type_is_zero_size(self.ccx, ty) {
                            let llty = type_of::type_of(self.ccx, ty);
                            return Ok(Const::new(C_null(llty), ty));
                        }

                        let substs = self.monomorphize(&substs);
                        let instance = Instance::new(def_id, substs);
                        MirConstContext::trans_def(self.ccx, instance, IndexVec::new())
                    }
                    mir::Literal::Promoted { index } => {
                        let mir = &self.mir.promoted[index];
                        MirConstContext::new(self.ccx, mir, self.substs, IndexVec::new()).trans()
                    }
                    mir::Literal::Value { value } => {
                        Ok(Const::from_constval(self.ccx, value, ty))
                    }
                }
            }
        };
        debug!("const_operand({:?} @ {:?}) = {:?}", operand, span,
               result.as_ref().ok());
        result
    }

    fn const_array(&self, array_ty: Ty<'tcx>, fields: &[ValueRef])
                   -> Const<'tcx>
    {
        let elem_ty = array_ty.builtin_index().unwrap_or_else(|| {
            bug!("bad array type {:?}", array_ty)
        });
        let llunitty = type_of::type_of(self.ccx, elem_ty);
        // If the array contains enums, an LLVM array won't work.
        let val = if fields.iter().all(|&f| val_ty(f) == llunitty) {
            C_array(llunitty, fields)
        } else {
            C_struct(self.ccx, fields, false)
        };
        Const::new(val, array_ty)
    }

    fn const_rvalue(&self, rvalue: &mir::Rvalue<'tcx>,
                    dest_ty: Ty<'tcx>, span: Span)
                    -> Result<Const<'tcx>, ConstEvalErr> {
        let tcx = self.ccx.tcx();
        debug!("const_rvalue({:?}: {:?} @ {:?})", rvalue, dest_ty, span);
        let val = match *rvalue {
            mir::Rvalue::Use(ref operand) => self.const_operand(operand, span)?,

            mir::Rvalue::Repeat(ref elem, ref count) => {
                let elem = self.const_operand(elem, span)?;
                let size = count.value.as_u64(tcx.sess.target.uint_type);
                let fields = vec![elem.llval; size as usize];
                self.const_array(dest_ty, &fields)
            }

            mir::Rvalue::Aggregate(ref kind, ref operands) => {
                // Make sure to evaluate all operands to
                // report as many errors as we possibly can.
                let mut fields = Vec::with_capacity(operands.len());
                let mut failure = Ok(());
                for operand in operands {
                    match self.const_operand(operand, span) {
                        Ok(val) => fields.push(val.llval),
                        Err(err) => if failure.is_ok() { failure = Err(err); }
                    }
                }
                failure?;

                match *kind {
                    mir::AggregateKind::Array => {
                        self.const_array(dest_ty, &fields)
                    }
                    mir::AggregateKind::Adt(..) |
                    mir::AggregateKind::Closure(..) |
                    mir::AggregateKind::Tuple => {
                        Const::new(trans_const(self.ccx, dest_ty, kind, &fields), dest_ty)
                    }
                }
            }

            mir::Rvalue::Cast(ref kind, ref source, cast_ty) => {
                let operand = self.const_operand(source, span)?;
                let cast_ty = self.monomorphize(&cast_ty);

                let val = match *kind {
                    mir::CastKind::ReifyFnPointer => {
                        match operand.ty.sty {
                            ty::TyFnDef(def_id, substs, _) => {
                                Callee::def(self.ccx, def_id, substs)
                                    .reify(self.ccx)
                            }
                            _ => {
                                span_bug!(span, "{} cannot be reified to a fn ptr",
                                          operand.ty)
                            }
                        }
                    }
                    mir::CastKind::UnsafeFnPointer => {
                        // this is a no-op at the LLVM level
                        operand.llval
                    }
                    mir::CastKind::Unsize => {
                        // unsize targets other than to a fat pointer currently
                        // can't be in constants.
                        assert!(common::type_is_fat_ptr(self.ccx, cast_ty));

                        let pointee_ty = operand.ty.builtin_deref(true, ty::NoPreference)
                            .expect("consts: unsizing got non-pointer type").ty;
                        let (base, old_info) = if !self.ccx.shared().type_is_sized(pointee_ty) {
                            // Normally, the source is a thin pointer and we are
                            // adding extra info to make a fat pointer. The exception
                            // is when we are upcasting an existing object fat pointer
                            // to use a different vtable. In that case, we want to
                            // load out the original data pointer so we can repackage
                            // it.
                            let (base, extra) = operand.get_fat_ptr();
                            (base, Some(extra))
                        } else {
                            (operand.llval, None)
                        };

                        let unsized_ty = cast_ty.builtin_deref(true, ty::NoPreference)
                            .expect("consts: unsizing got non-pointer target type").ty;
                        let ptr_ty = type_of::in_memory_type_of(self.ccx, unsized_ty).ptr_to();
                        let base = consts::ptrcast(base, ptr_ty);
                        let info = base::unsized_info(self.ccx, pointee_ty,
                                                      unsized_ty, old_info);

                        if old_info.is_none() {
                            let prev_const = self.ccx.const_unsized().borrow_mut()
                                                     .insert(base, operand.llval);
                            assert!(prev_const.is_none() || prev_const == Some(operand.llval));
                        }
                        assert_eq!(abi::FAT_PTR_ADDR, 0);
                        assert_eq!(abi::FAT_PTR_EXTRA, 1);
                        C_struct(self.ccx, &[base, info], false)
                    }
                    mir::CastKind::Misc if common::type_is_immediate(self.ccx, operand.ty) => {
                        debug_assert!(common::type_is_immediate(self.ccx, cast_ty));
                        let r_t_in = CastTy::from_ty(operand.ty).expect("bad input type for cast");
                        let r_t_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                        let ll_t_out = type_of::immediate_type_of(self.ccx, cast_ty);
                        let llval = operand.llval;
                        let signed = if let CastTy::Int(IntTy::CEnum) = r_t_in {
                            let l = self.ccx.layout_of(operand.ty);
                            adt::is_discr_signed(&l)
                        } else {
                            operand.ty.is_signed()
                        };

                        unsafe {
                            match (r_t_in, r_t_out) {
                                (CastTy::Int(_), CastTy::Int(_)) => {
                                    let s = signed as llvm::Bool;
                                    llvm::LLVMConstIntCast(llval, ll_t_out.to_ref(), s)
                                }
                                (CastTy::Int(_), CastTy::Float) => {
                                    if signed {
                                        llvm::LLVMConstSIToFP(llval, ll_t_out.to_ref())
                                    } else {
                                        llvm::LLVMConstUIToFP(llval, ll_t_out.to_ref())
                                    }
                                }
                                (CastTy::Float, CastTy::Float) => {
                                    llvm::LLVMConstFPCast(llval, ll_t_out.to_ref())
                                }
                                (CastTy::Float, CastTy::Int(IntTy::I)) => {
                                    llvm::LLVMConstFPToSI(llval, ll_t_out.to_ref())
                                }
                                (CastTy::Float, CastTy::Int(_)) => {
                                    llvm::LLVMConstFPToUI(llval, ll_t_out.to_ref())
                                }
                                (CastTy::Ptr(_), CastTy::Ptr(_)) |
                                (CastTy::FnPtr, CastTy::Ptr(_)) |
                                (CastTy::RPtr(_), CastTy::Ptr(_)) => {
                                    consts::ptrcast(llval, ll_t_out)
                                }
                                (CastTy::Int(_), CastTy::Ptr(_)) => {
                                    llvm::LLVMConstIntToPtr(llval, ll_t_out.to_ref())
                                }
                                (CastTy::Ptr(_), CastTy::Int(_)) |
                                (CastTy::FnPtr, CastTy::Int(_)) => {
                                    llvm::LLVMConstPtrToInt(llval, ll_t_out.to_ref())
                                }
                                _ => bug!("unsupported cast: {:?} to {:?}", operand.ty, cast_ty)
                            }
                        }
                    }
                    mir::CastKind::Misc => { // Casts from a fat-ptr.
                        let ll_cast_ty = type_of::immediate_type_of(self.ccx, cast_ty);
                        let ll_from_ty = type_of::immediate_type_of(self.ccx, operand.ty);
                        if common::type_is_fat_ptr(self.ccx, operand.ty) {
                            let (data_ptr, meta_ptr) = operand.get_fat_ptr();
                            if common::type_is_fat_ptr(self.ccx, cast_ty) {
                                let ll_cft = ll_cast_ty.field_types();
                                let ll_fft = ll_from_ty.field_types();
                                let data_cast = consts::ptrcast(data_ptr, ll_cft[0]);
                                assert_eq!(ll_cft[1].kind(), ll_fft[1].kind());
                                C_struct(self.ccx, &[data_cast, meta_ptr], false)
                            } else { // cast to thin-ptr
                                // Cast of fat-ptr to thin-ptr is an extraction of data-ptr and
                                // pointer-cast of that pointer to desired pointer type.
                                consts::ptrcast(data_ptr, ll_cast_ty)
                            }
                        } else {
                            bug!("Unexpected non-fat-pointer operand")
                        }
                    }
                };
                Const::new(val, cast_ty)
            }

            mir::Rvalue::Ref(_, bk, ref lvalue) => {
                let tr_lvalue = self.const_lvalue(lvalue, span)?;

                let ty = tr_lvalue.ty;
                let ref_ty = tcx.mk_ref(tcx.mk_region(ty::ReErased),
                    ty::TypeAndMut { ty: ty, mutbl: bk.to_mutbl_lossy() });

                let base = match tr_lvalue.base {
                    Base::Value(llval) => {
                        // FIXME: may be wrong for &*(&simd_vec as &fmt::Debug)
                        let align = if self.ccx.shared().type_is_sized(ty) {
                            type_of::align_of(self.ccx, ty)
                        } else {
                            self.ccx.tcx().data_layout.pointer_align.abi() as machine::llalign
                        };
                        if bk == mir::BorrowKind::Mut {
                            consts::addr_of_mut(self.ccx, llval, align, "ref_mut")
                        } else {
                            consts::addr_of(self.ccx, llval, align, "ref")
                        }
                    }
                    Base::Str(llval) |
                    Base::Static(llval) => llval
                };

                let ptr = if self.ccx.shared().type_is_sized(ty) {
                    base
                } else {
                    C_struct(self.ccx, &[base, tr_lvalue.llextra], false)
                };
                Const::new(ptr, ref_ty)
            }

            mir::Rvalue::Len(ref lvalue) => {
                let tr_lvalue = self.const_lvalue(lvalue, span)?;
                Const::new(tr_lvalue.len(self.ccx), tcx.types.usize)
            }

            mir::Rvalue::BinaryOp(op, ref lhs, ref rhs) => {
                let lhs = self.const_operand(lhs, span)?;
                let rhs = self.const_operand(rhs, span)?;
                let ty = lhs.ty;
                let binop_ty = op.ty(tcx, lhs.ty, rhs.ty);
                let (lhs, rhs) = (lhs.llval, rhs.llval);
                Const::new(const_scalar_binop(op, lhs, rhs, ty), binop_ty)
            }

            mir::Rvalue::CheckedBinaryOp(op, ref lhs, ref rhs) => {
                let lhs = self.const_operand(lhs, span)?;
                let rhs = self.const_operand(rhs, span)?;
                let ty = lhs.ty;
                let val_ty = op.ty(tcx, lhs.ty, rhs.ty);
                let binop_ty = tcx.intern_tup(&[val_ty, tcx.types.bool]);
                let (lhs, rhs) = (lhs.llval, rhs.llval);
                assert!(!ty.is_fp());

                match const_scalar_checked_binop(tcx, op, lhs, rhs, ty) {
                    Some((llval, of)) => {
                        let llof = C_bool(self.ccx, of);
                        Const::new(C_struct(self.ccx, &[llval, llof], false), binop_ty)
                    }
                    None => {
                        span_bug!(span, "{:?} got non-integer operands: {:?} and {:?}",
                                  rvalue, Value(lhs), Value(rhs));
                    }
                }
            }

            mir::Rvalue::UnaryOp(op, ref operand) => {
                let operand = self.const_operand(operand, span)?;
                let lloperand = operand.llval;
                let llval = match op {
                    mir::UnOp::Not => {
                        unsafe {
                            llvm::LLVMConstNot(lloperand)
                        }
                    }
                    mir::UnOp::Neg => {
                        let is_float = operand.ty.is_fp();
                        unsafe {
                            if is_float {
                                llvm::LLVMConstFNeg(lloperand)
                            } else {
                                llvm::LLVMConstNeg(lloperand)
                            }
                        }
                    }
                };
                Const::new(llval, operand.ty)
            }

            _ => span_bug!(span, "{:?} in constant", rvalue)
        };

        debug!("const_rvalue({:?}: {:?} @ {:?}) = {:?}", rvalue, dest_ty, span, val);

        Ok(val)
    }

}

fn to_const_int(value: ValueRef, t: Ty, tcx: TyCtxt) -> Option<ConstInt> {
    match t.sty {
        ty::TyInt(int_type) => const_to_opt_u128(value, true)
            .and_then(|input| ConstInt::new_signed(input as i128, int_type,
                                                   tcx.sess.target.int_type)),
        ty::TyUint(uint_type) => const_to_opt_u128(value, false)
            .and_then(|input| ConstInt::new_unsigned(input, uint_type,
                                                     tcx.sess.target.uint_type)),
        _ => None

    }
}

pub fn const_scalar_binop(op: mir::BinOp,
                          lhs: ValueRef,
                          rhs: ValueRef,
                          input_ty: Ty) -> ValueRef {
    assert!(!input_ty.is_simd());
    let is_float = input_ty.is_fp();
    let signed = input_ty.is_signed();

    unsafe {
        match op {
            mir::BinOp::Add if is_float => llvm::LLVMConstFAdd(lhs, rhs),
            mir::BinOp::Add             => llvm::LLVMConstAdd(lhs, rhs),

            mir::BinOp::Sub if is_float => llvm::LLVMConstFSub(lhs, rhs),
            mir::BinOp::Sub             => llvm::LLVMConstSub(lhs, rhs),

            mir::BinOp::Mul if is_float => llvm::LLVMConstFMul(lhs, rhs),
            mir::BinOp::Mul             => llvm::LLVMConstMul(lhs, rhs),

            mir::BinOp::Div if is_float => llvm::LLVMConstFDiv(lhs, rhs),
            mir::BinOp::Div if signed   => llvm::LLVMConstSDiv(lhs, rhs),
            mir::BinOp::Div             => llvm::LLVMConstUDiv(lhs, rhs),

            mir::BinOp::Rem if is_float => llvm::LLVMConstFRem(lhs, rhs),
            mir::BinOp::Rem if signed   => llvm::LLVMConstSRem(lhs, rhs),
            mir::BinOp::Rem             => llvm::LLVMConstURem(lhs, rhs),

            mir::BinOp::BitXor => llvm::LLVMConstXor(lhs, rhs),
            mir::BinOp::BitAnd => llvm::LLVMConstAnd(lhs, rhs),
            mir::BinOp::BitOr  => llvm::LLVMConstOr(lhs, rhs),
            mir::BinOp::Shl    => {
                let rhs = base::cast_shift_const_rhs(op.to_hir_binop(), lhs, rhs);
                llvm::LLVMConstShl(lhs, rhs)
            }
            mir::BinOp::Shr    => {
                let rhs = base::cast_shift_const_rhs(op.to_hir_binop(), lhs, rhs);
                if signed { llvm::LLVMConstAShr(lhs, rhs) }
                else      { llvm::LLVMConstLShr(lhs, rhs) }
            }
            mir::BinOp::Eq | mir::BinOp::Ne |
            mir::BinOp::Lt | mir::BinOp::Le |
            mir::BinOp::Gt | mir::BinOp::Ge => {
                if is_float {
                    let cmp = base::bin_op_to_fcmp_predicate(op.to_hir_binop());
                    llvm::LLVMConstFCmp(cmp, lhs, rhs)
                } else {
                    let cmp = base::bin_op_to_icmp_predicate(op.to_hir_binop(),
                                                                signed);
                    llvm::LLVMConstICmp(cmp, lhs, rhs)
                }
            }
        }
    }
}

pub fn const_scalar_checked_binop<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                            op: mir::BinOp,
                                            lllhs: ValueRef,
                                            llrhs: ValueRef,
                                            input_ty: Ty<'tcx>)
                                            -> Option<(ValueRef, bool)> {
    if let (Some(lhs), Some(rhs)) = (to_const_int(lllhs, input_ty, tcx),
                                     to_const_int(llrhs, input_ty, tcx)) {
        let result = match op {
            mir::BinOp::Add => lhs + rhs,
            mir::BinOp::Sub => lhs - rhs,
            mir::BinOp::Mul => lhs * rhs,
            mir::BinOp::Shl => lhs << rhs,
            mir::BinOp::Shr => lhs >> rhs,
            _ => {
                bug!("Operator `{:?}` is not a checkable operator", op)
            }
        };

        let of = match result {
            Ok(_) => false,
            Err(ConstMathErr::Overflow(_)) |
            Err(ConstMathErr::ShiftNegative) => true,
            Err(err) => {
                bug!("Operator `{:?}` on `{:?}` and `{:?}` errored: {}",
                     op, lhs, rhs, err.description());
            }
        };

        Some((const_scalar_binop(op, lllhs, llrhs, input_ty), of))
    } else {
        None
    }
}

impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn trans_constant(&mut self,
                          bcx: &Builder<'a, 'tcx>,
                          constant: &mir::Constant<'tcx>)
                          -> Const<'tcx>
    {
        debug!("trans_constant({:?})", constant);
        let ty = self.monomorphize(&constant.ty);
        let result = match constant.literal.clone() {
            mir::Literal::Item { def_id, substs } => {
                // Shortcut for zero-sized types, including function item
                // types, which would not work with MirConstContext.
                if common::type_is_zero_size(bcx.ccx, ty) {
                    let llty = type_of::type_of(bcx.ccx, ty);
                    return Const::new(C_null(llty), ty);
                }

                let substs = self.monomorphize(&substs);
                let instance = Instance::new(def_id, substs);
                MirConstContext::trans_def(bcx.ccx, instance, IndexVec::new())
            }
            mir::Literal::Promoted { index } => {
                let mir = &self.mir.promoted[index];
                MirConstContext::new(bcx.ccx, mir, self.param_substs, IndexVec::new()).trans()
            }
            mir::Literal::Value { value } => {
                Ok(Const::from_constval(bcx.ccx, value, ty))
            }
        };

        let result = result.unwrap_or_else(|_| {
            // We've errored, so we don't have to produce working code.
            let llty = type_of::type_of(bcx.ccx, ty);
            Const::new(C_undef(llty), ty)
        });

        debug!("trans_constant({:?}) = {:?}", constant, result);
        result
    }
}


pub fn trans_static_initializer(ccx: &CrateContext, def_id: DefId)
                                -> Result<ValueRef, ConstEvalErr> {
    let instance = Instance::mono(ccx.shared(), def_id);
    MirConstContext::trans_def(ccx, instance, IndexVec::new()).map(|c| c.llval)
}

/// Construct a constant value, suitable for initializing a
/// GlobalVariable, given a case and constant values for its fields.
/// Note that this may have a different LLVM type (and different
/// alignment!) from the representation's `type_of`, so it needs a
/// pointer cast before use.
///
/// The LLVM type system does not directly support unions, and only
/// pointers can be bitcast, so a constant (and, by extension, the
/// GlobalVariable initialized by it) will have a type that can vary
/// depending on which case of an enum it is.
///
/// To understand the alignment situation, consider `enum E { V64(u64),
/// V32(u32, u32) }` on Windows.  The type has 8-byte alignment to
/// accommodate the u64, but `V32(x, y)` would have LLVM type `{i32,
/// i32, i32}`, which is 4-byte aligned.
///
/// Currently the returned value has the same size as the type, but
/// this could be changed in the future to avoid allocating unnecessary
/// space after values of shorter-than-maximum cases.
fn trans_const<'a, 'tcx>(
    ccx: &CrateContext<'a, 'tcx>,
    t: Ty<'tcx>,
    kind: &mir::AggregateKind,
    vals: &[ValueRef]
) -> ValueRef {
    let l = ccx.layout_of(t);
    let dl = &ccx.tcx().data_layout;
    let variant_index = match *kind {
        mir::AggregateKind::Adt(_, index, _, _) => index,
        _ => 0,
    };
    match *l {
        layout::CEnum { discr: d, min, max, .. } => {
            let discr = match *kind {
                mir::AggregateKind::Adt(adt_def, _, _, _) => {
                    Disr::from(adt_def.variants[variant_index].disr_val)
                },
                _ => Disr(0),
            };
            assert_eq!(vals.len(), 0);
            adt::assert_discr_in_range(Disr(min), Disr(max), discr);
            C_integral(Type::from_integer(ccx, d), discr.0, true)
        }
        layout::General { discr: d, ref variants, .. } => {
            let variant = &variants[variant_index];
            let lldiscr = C_integral(Type::from_integer(ccx, d), variant_index as u64, true);
            let mut vals_with_discr = vec![lldiscr];
            vals_with_discr.extend_from_slice(vals);
            let mut contents = build_const_struct(ccx, &variant, &vals_with_discr[..]);
            let needed_padding = l.size(dl).bytes() - variant.stride().bytes();
            if needed_padding > 0 {
                contents.push(padding(ccx, needed_padding));
            }
            C_struct(ccx, &contents[..], false)
        }
        layout::UntaggedUnion { ref variants, .. }=> {
            assert_eq!(variant_index, 0);
            let contents = build_const_union(ccx, variants, vals[0]);
            C_struct(ccx, &contents, variants.packed)
        }
        layout::Univariant { ref variant, .. } => {
            assert_eq!(variant_index, 0);
            let contents = build_const_struct(ccx, &variant, vals);
            C_struct(ccx, &contents[..], variant.packed)
        }
        layout::Vector { .. } => {
            C_vector(vals)
        }
        layout::RawNullablePointer { nndiscr, .. } => {
            let nnty = adt::compute_fields(ccx, t, nndiscr as usize, false)[0];
            if variant_index as u64 == nndiscr {
                assert_eq!(vals.len(), 1);
                vals[0]
            } else {
                C_null(type_of::sizing_type_of(ccx, nnty))
            }
        }
        layout::StructWrappedNullablePointer { ref nonnull, nndiscr, .. } => {
            if variant_index as u64 == nndiscr {
                C_struct(ccx, &build_const_struct(ccx, &nonnull, vals), false)
            } else {
                let fields = adt::compute_fields(ccx, t, nndiscr as usize, false);
                let vals = fields.iter().map(|&ty| {
                    // Always use null even if it's not the `discrfield`th
                    // field; see #8506.
                    C_null(type_of::sizing_type_of(ccx, ty))
                }).collect::<Vec<ValueRef>>();
                C_struct(ccx, &build_const_struct(ccx, &nonnull, &vals[..]), false)
            }
        }
        _ => bug!("trans_const: cannot handle type {} repreented as {:#?}", t, l)
    }
}

/// Building structs is a little complicated, because we might need to
/// insert padding if a field's value is less aligned than its type.
///
/// Continuing the example from `trans_const`, a value of type `(u32,
/// E)` should have the `E` at offset 8, but if that field's
/// initializer is 4-byte aligned then simply translating the tuple as
/// a two-element struct will locate it at offset 4, and accesses to it
/// will read the wrong memory.
fn build_const_struct<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                st: &layout::Struct,
                                vals: &[ValueRef])
                                -> Vec<ValueRef> {
    assert_eq!(vals.len(), st.offsets.len());

    if vals.len() == 0 {
        return Vec::new();
    }

    // offset of current value
    let mut offset = 0;
    let mut cfields = Vec::new();
    cfields.reserve(st.offsets.len()*2);

    let parts = st.field_index_by_increasing_offset().map(|i| {
        (&vals[i], st.offsets[i].bytes())
    });
    for (&val, target_offset) in parts {
        if offset < target_offset {
            cfields.push(padding(ccx, target_offset - offset));
            offset = target_offset;
        }
        assert!(!is_undef(val));
        cfields.push(val);
        offset += machine::llsize_of_alloc(ccx, val_ty(val));
    }

    if offset < st.stride().bytes() {
        cfields.push(padding(ccx, st.stride().bytes() - offset));
    }

    cfields
}

fn build_const_union<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                               un: &layout::Union,
                               field_val: ValueRef)
                               -> Vec<ValueRef> {
    let mut cfields = vec![field_val];

    let offset = machine::llsize_of_alloc(ccx, val_ty(field_val));
    let size = un.stride().bytes();
    if offset != size {
        cfields.push(padding(ccx, size - offset));
    }

    cfields
}

fn padding(ccx: &CrateContext, size: u64) -> ValueRef {
    C_undef(Type::array(&Type::i8(ccx), size))
}
