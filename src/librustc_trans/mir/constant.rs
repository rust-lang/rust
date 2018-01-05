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
use rustc::middle::const_val::{ConstEvalErr, ConstVal, ErrKind};
use rustc_const_math::ConstInt::*;
use rustc_const_math::{ConstInt, ConstMathErr, MAX_F32_PLUS_HALF_ULP};
use rustc::hir::def_id::DefId;
use rustc::infer::TransNormalize;
use rustc::traits;
use rustc::mir;
use rustc::mir::tcx::PlaceTy;
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc::ty::layout::{self, LayoutOf, Size};
use rustc::ty::cast::{CastTy, IntTy};
use rustc::ty::subst::{Kind, Substs};
use rustc_apfloat::{ieee, Float, Status};
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use base;
use abi::{self, Abi};
use callee;
use builder::Builder;
use common::{self, CodegenCx, const_get_elt, val_ty};
use common::{C_array, C_bool, C_bytes, C_int, C_uint, C_uint_big, C_u32, C_u64};
use common::{C_null, C_struct, C_str_slice, C_undef, C_usize, C_vector, C_fat_ptr};
use common::const_to_opt_u128;
use consts;
use type_of::LayoutLlvmExt;
use type_::Type;
use value::Value;

use syntax_pos::Span;
use syntax::ast;

use std::fmt;
use std::ptr;

use super::operand::{OperandRef, OperandValue};
use super::FunctionCx;

/// A sized constant rvalue.
/// The LLVM type might not be the same for a single Rust type,
/// e.g. each enum variant would have its own LLVM struct type.
#[derive(Copy, Clone)]
pub struct Const<'tcx> {
    pub llval: ValueRef,
    pub ty: Ty<'tcx>
}

impl<'a, 'tcx> Const<'tcx> {
    pub fn new(llval: ValueRef, ty: Ty<'tcx>) -> Const<'tcx> {
        Const {
            llval,
            ty,
        }
    }

    pub fn from_constint(cx: &CodegenCx<'a, 'tcx>, ci: &ConstInt) -> Const<'tcx> {
        let tcx = cx.tcx;
        let (llval, ty) = match *ci {
            I8(v) => (C_int(Type::i8(cx), v as i64), tcx.types.i8),
            I16(v) => (C_int(Type::i16(cx), v as i64), tcx.types.i16),
            I32(v) => (C_int(Type::i32(cx), v as i64), tcx.types.i32),
            I64(v) => (C_int(Type::i64(cx), v as i64), tcx.types.i64),
            I128(v) => (C_uint_big(Type::i128(cx), v as u128), tcx.types.i128),
            Isize(v) => (C_int(Type::isize(cx), v.as_i64()), tcx.types.isize),
            U8(v) => (C_uint(Type::i8(cx), v as u64), tcx.types.u8),
            U16(v) => (C_uint(Type::i16(cx), v as u64), tcx.types.u16),
            U32(v) => (C_uint(Type::i32(cx), v as u64), tcx.types.u32),
            U64(v) => (C_uint(Type::i64(cx), v), tcx.types.u64),
            U128(v) => (C_uint_big(Type::i128(cx), v), tcx.types.u128),
            Usize(v) => (C_uint(Type::isize(cx), v.as_u64()), tcx.types.usize),
        };
        Const { llval: llval, ty: ty }
    }

    /// Translate ConstVal into a LLVM constant value.
    pub fn from_constval(cx: &CodegenCx<'a, 'tcx>,
                         cv: &ConstVal,
                         ty: Ty<'tcx>)
                         -> Const<'tcx> {
        let llty = cx.layout_of(ty).llvm_type(cx);
        let val = match *cv {
            ConstVal::Float(v) => {
                let bits = match v.ty {
                    ast::FloatTy::F32 => C_u32(cx, v.bits as u32),
                    ast::FloatTy::F64 => C_u64(cx, v.bits as u64)
                };
                consts::bitcast(bits, llty)
            }
            ConstVal::Bool(v) => C_bool(cx, v),
            ConstVal::Integral(ref i) => return Const::from_constint(cx, i),
            ConstVal::Str(ref v) => C_str_slice(cx, v.clone()),
            ConstVal::ByteStr(v) => {
                consts::addr_of(cx, C_bytes(cx, v.data), cx.align_of(ty), "byte_str")
            }
            ConstVal::Char(c) => C_uint(Type::char(cx), c as u64),
            ConstVal::Function(..) => C_undef(llty),
            ConstVal::Variant(_) |
            ConstVal::Aggregate(..) |
            ConstVal::Unevaluated(..) => {
                bug!("MIR must not use `{:?}` (aggregates are expanded to MIR rvalues)", cv)
            }
        };

        assert!(!ty.has_erasable_regions());

        Const::new(val, ty)
    }

    fn get_field(&self, cx: &CodegenCx<'a, 'tcx>, i: usize) -> ValueRef {
        let layout = cx.layout_of(self.ty);
        let field = layout.field(cx, i);
        if field.is_zst() {
            return C_undef(field.immediate_llvm_type(cx));
        }
        let offset = layout.fields.offset(i);
        match layout.abi {
            layout::Abi::Scalar(_) |
            layout::Abi::ScalarPair(..) |
            layout::Abi::Vector { .. }
                if offset.bytes() == 0 && field.size == layout.size => self.llval,

            layout::Abi::ScalarPair(ref a, ref b) => {
                if offset.bytes() == 0 {
                    assert_eq!(field.size, a.value.size(cx));
                    const_get_elt(self.llval, 0)
                } else {
                    assert_eq!(offset, a.value.size(cx)
                        .abi_align(b.value.align(cx)));
                    assert_eq!(field.size, b.value.size(cx));
                    const_get_elt(self.llval, 1)
                }
            }
            _ => {
                const_get_elt(self.llval, layout.llvm_field_index(i))
            }
        }
    }

    fn get_pair(&self, cx: &CodegenCx<'a, 'tcx>) -> (ValueRef, ValueRef) {
        (self.get_field(cx, 0), self.get_field(cx, 1))
    }

    fn get_fat_ptr(&self, cx: &CodegenCx<'a, 'tcx>) -> (ValueRef, ValueRef) {
        assert_eq!(abi::FAT_PTR_ADDR, 0);
        assert_eq!(abi::FAT_PTR_EXTRA, 1);
        self.get_pair(cx)
    }

    fn as_place(&self) -> ConstPlace<'tcx> {
        ConstPlace {
            base: Base::Value(self.llval),
            llextra: ptr::null_mut(),
            ty: self.ty
        }
    }

    pub fn to_operand(&self, cx: &CodegenCx<'a, 'tcx>) -> OperandRef<'tcx> {
        let layout = cx.layout_of(self.ty);
        let llty = layout.immediate_llvm_type(cx);
        let llvalty = val_ty(self.llval);

        let val = if llty == llvalty && layout.is_llvm_scalar_pair() {
            OperandValue::Pair(
                const_get_elt(self.llval, 0),
                const_get_elt(self.llval, 1))
        } else if llty == llvalty && layout.is_llvm_immediate() {
            // If the types match, we can use the value directly.
            OperandValue::Immediate(self.llval)
        } else {
            // Otherwise, or if the value is not immediate, we create
            // a constant LLVM global and cast its address if necessary.
            let align = cx.align_of(self.ty);
            let ptr = consts::addr_of(cx, self.llval, align, "const");
            OperandValue::Ref(consts::ptrcast(ptr, layout.llvm_type(cx).ptr_to()),
                              layout.align)
        };

        OperandRef {
            val,
            layout
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

/// An place as seen from a constant.
#[derive(Copy, Clone)]
struct ConstPlace<'tcx> {
    base: Base,
    llextra: ValueRef,
    ty: Ty<'tcx>
}

impl<'tcx> ConstPlace<'tcx> {
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

    pub fn len<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> ValueRef {
        match self.ty.sty {
            ty::TyArray(_, n) => {
                C_usize(cx, n.val.to_const_int().unwrap().to_u64().unwrap())
            }
            ty::TySlice(_) | ty::TyStr => {
                assert!(self.llextra != ptr::null_mut());
                self.llextra
            }
            _ => bug!("unexpected type `{}` in ConstPlace::len", self.ty)
        }
    }
}

/// Machinery for translating a constant's MIR to LLVM values.
/// FIXME(eddyb) use miri and lower its allocations to LLVM.
struct MirConstContext<'a, 'tcx: 'a> {
    cx: &'a CodegenCx<'a, 'tcx>,
    mir: &'a mir::Mir<'tcx>,

    /// Type parameters for const fn and associated constants.
    substs: &'tcx Substs<'tcx>,

    /// Values of locals in a constant or const fn.
    locals: IndexVec<mir::Local, Option<Result<Const<'tcx>, ConstEvalErr<'tcx>>>>
}

fn add_err<'tcx, U, V>(failure: &mut Result<U, ConstEvalErr<'tcx>>,
                       value: &Result<V, ConstEvalErr<'tcx>>)
{
    if let &Err(ref err) = value {
        if failure.is_ok() {
            *failure = Err(err.clone());
        }
    }
}

impl<'a, 'tcx> MirConstContext<'a, 'tcx> {
    fn new(cx: &'a CodegenCx<'a, 'tcx>,
           mir: &'a mir::Mir<'tcx>,
           substs: &'tcx Substs<'tcx>,
           args: IndexVec<mir::Local, Result<Const<'tcx>, ConstEvalErr<'tcx>>>)
           -> MirConstContext<'a, 'tcx> {
        let mut context = MirConstContext {
            cx,
            mir,
            substs,
            locals: (0..mir.local_decls.len()).map(|_| None).collect(),
        };
        for (i, arg) in args.into_iter().enumerate() {
            // Locals after local 0 are the function arguments
            let index = mir::Local::new(i + 1);
            context.locals[index] = Some(arg);
        }
        context
    }

    fn trans_def(cx: &'a CodegenCx<'a, 'tcx>,
                 def_id: DefId,
                 substs: &'tcx Substs<'tcx>,
                 args: IndexVec<mir::Local, Result<Const<'tcx>, ConstEvalErr<'tcx>>>)
                 -> Result<Const<'tcx>, ConstEvalErr<'tcx>> {
        let instance = ty::Instance::resolve(cx.tcx,
                                             ty::ParamEnv::empty(traits::Reveal::All),
                                             def_id,
                                             substs).unwrap();
        let mir = cx.tcx.instance_mir(instance.def);
        MirConstContext::new(cx, &mir, instance.substs, args).trans()
    }

    fn monomorphize<T>(&self, value: &T) -> T
        where T: TransNormalize<'tcx>
    {
        self.cx.tcx.trans_apply_param_substs(self.substs, value)
    }

    fn trans(&mut self) -> Result<Const<'tcx>, ConstEvalErr<'tcx>> {
        let tcx = self.cx.tcx;
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
                        let value = self.const_rvalue(rvalue, ty, span);
                        add_err(&mut failure, &value);
                        self.store(dest, value, span);
                    }
                    mir::StatementKind::StorageLive(_) |
                    mir::StatementKind::StorageDead(_) |
                    mir::StatementKind::Validate(..) |
                    mir::StatementKind::EndRegion(_) |
                    mir::StatementKind::Nop => {}
                    mir::StatementKind::InlineAsm { .. } |
                    mir::StatementKind::SetDiscriminant{ .. } => {
                        span_bug!(span, "{:?} should not appear in constants?", statement.kind);
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
                    return self.locals[mir::RETURN_PLACE].clone().unwrap_or_else(|| {
                        span_bug!(span, "no returned value in constant");
                    });
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
                            mir::AssertMessage::GeneratorResumedAfterReturn |
                            mir::AssertMessage::GeneratorResumedAfterPanic =>
                                span_bug!(span, "{:?} should not appear in constants?", msg),
                        };

                        let err = ConstEvalErr { span: span, kind: err };
                        err.report(tcx, span, "expression");
                        failure = Err(err);
                    }
                    target
                }

                mir::TerminatorKind::Call { ref func, ref args, ref destination, .. } => {
                    let fn_ty = func.ty(self.mir, tcx);
                    let fn_ty = self.monomorphize(&fn_ty);
                    let (def_id, substs) = match fn_ty.sty {
                        ty::TyFnDef(def_id, substs) => (def_id, substs),
                        _ => span_bug!(span, "calling {:?} (of type {}) in constant",
                                       func, fn_ty)
                    };

                    let mut arg_vals = IndexVec::with_capacity(args.len());
                    for arg in args {
                        let arg_val = self.const_operand(arg, span);
                        add_err(&mut failure, &arg_val);
                        arg_vals.push(arg_val);
                    }
                    if let Some((ref dest, target)) = *destination {
                        let result = if fn_ty.fn_sig(tcx).abi() == Abi::RustIntrinsic {
                            match &tcx.item_name(def_id)[..] {
                                "size_of" => {
                                    let llval = C_usize(self.cx,
                                        self.cx.size_of(substs.type_at(0)).bytes());
                                    Ok(Const::new(llval, tcx.types.usize))
                                }
                                "min_align_of" => {
                                    let llval = C_usize(self.cx,
                                        self.cx.align_of(substs.type_at(0)).abi());
                                    Ok(Const::new(llval, tcx.types.usize))
                                }
                                _ => span_bug!(span, "{:?} in constant", terminator.kind)
                            }
                        } else if let Some((op, is_checked)) = self.is_binop_lang_item(def_id) {
                            (||{
                                assert_eq!(arg_vals.len(), 2);
                                let rhs = arg_vals.pop().unwrap()?;
                                let lhs = arg_vals.pop().unwrap()?;
                                if !is_checked {
                                    let binop_ty = op.ty(tcx, lhs.ty, rhs.ty);
                                    let (lhs, rhs) = (lhs.llval, rhs.llval);
                                    Ok(Const::new(const_scalar_binop(op, lhs, rhs, binop_ty),
                                                  binop_ty))
                                } else {
                                    let ty = lhs.ty;
                                    let val_ty = op.ty(tcx, lhs.ty, rhs.ty);
                                    let binop_ty = tcx.intern_tup(&[val_ty, tcx.types.bool], false);
                                    let (lhs, rhs) = (lhs.llval, rhs.llval);
                                    assert!(!ty.is_fp());

                                    match const_scalar_checked_binop(tcx, op, lhs, rhs, ty) {
                                        Some((llval, of)) => {
                                            Ok(trans_const_adt(
                                                self.cx,
                                                binop_ty,
                                                &mir::AggregateKind::Tuple,
                                                &[
                                                    Const::new(llval, val_ty),
                                                    Const::new(C_bool(self.cx, of), tcx.types.bool)
                                                ]))
                                        }
                                        None => {
                                            span_bug!(span,
                                                "{:?} got non-integer operands: {:?} and {:?}",
                                                op, Value(lhs), Value(rhs));
                                        }
                                    }
                                }
                            })()
                        } else {
                            MirConstContext::trans_def(self.cx, def_id, substs, arg_vals)
                        };
                        add_err(&mut failure, &result);
                        self.store(dest, result, span);
                        target
                    } else {
                        span_bug!(span, "diverging {:?} in constant", terminator.kind);
                    }
                }
                _ => span_bug!(span, "{:?} in constant", terminator.kind)
            };
        }
    }

    fn is_binop_lang_item(&mut self, def_id: DefId) -> Option<(mir::BinOp, bool)> {
        let tcx = self.cx.tcx;
        let items = tcx.lang_items();
        let def_id = Some(def_id);
        if items.i128_add_fn() == def_id { Some((mir::BinOp::Add, false)) }
        else if items.u128_add_fn() == def_id { Some((mir::BinOp::Add, false)) }
        else if items.i128_sub_fn() == def_id { Some((mir::BinOp::Sub, false)) }
        else if items.u128_sub_fn() == def_id { Some((mir::BinOp::Sub, false)) }
        else if items.i128_mul_fn() == def_id { Some((mir::BinOp::Mul, false)) }
        else if items.u128_mul_fn() == def_id { Some((mir::BinOp::Mul, false)) }
        else if items.i128_div_fn() == def_id { Some((mir::BinOp::Div, false)) }
        else if items.u128_div_fn() == def_id { Some((mir::BinOp::Div, false)) }
        else if items.i128_rem_fn() == def_id { Some((mir::BinOp::Rem, false)) }
        else if items.u128_rem_fn() == def_id { Some((mir::BinOp::Rem, false)) }
        else if items.i128_shl_fn() == def_id { Some((mir::BinOp::Shl, false)) }
        else if items.u128_shl_fn() == def_id { Some((mir::BinOp::Shl, false)) }
        else if items.i128_shr_fn() == def_id { Some((mir::BinOp::Shr, false)) }
        else if items.u128_shr_fn() == def_id { Some((mir::BinOp::Shr, false)) }
        else if items.i128_addo_fn() == def_id { Some((mir::BinOp::Add, true)) }
        else if items.u128_addo_fn() == def_id { Some((mir::BinOp::Add, true)) }
        else if items.i128_subo_fn() == def_id { Some((mir::BinOp::Sub, true)) }
        else if items.u128_subo_fn() == def_id { Some((mir::BinOp::Sub, true)) }
        else if items.i128_mulo_fn() == def_id { Some((mir::BinOp::Mul, true)) }
        else if items.u128_mulo_fn() == def_id { Some((mir::BinOp::Mul, true)) }
        else if items.i128_shlo_fn() == def_id { Some((mir::BinOp::Shl, true)) }
        else if items.u128_shlo_fn() == def_id { Some((mir::BinOp::Shl, true)) }
        else if items.i128_shro_fn() == def_id { Some((mir::BinOp::Shr, true)) }
        else if items.u128_shro_fn() == def_id { Some((mir::BinOp::Shr, true)) }
        else { None }
    }

    fn store(&mut self,
             dest: &mir::Place<'tcx>,
             value: Result<Const<'tcx>, ConstEvalErr<'tcx>>,
             span: Span) {
        if let mir::Place::Local(index) = *dest {
            self.locals[index] = Some(value);
        } else {
            span_bug!(span, "assignment to {:?} in constant", dest);
        }
    }

    fn const_place(&self, place: &mir::Place<'tcx>, span: Span)
                    -> Result<ConstPlace<'tcx>, ConstEvalErr<'tcx>> {
        let tcx = self.cx.tcx;

        if let mir::Place::Local(index) = *place {
            return self.locals[index].clone().unwrap_or_else(|| {
                span_bug!(span, "{:?} not initialized", place)
            }).map(|v| v.as_place());
        }

        let place = match *place {
            mir::Place::Local(_)  => bug!(), // handled above
            mir::Place::Static(box mir::Static { def_id, ty }) => {
                ConstPlace {
                    base: Base::Static(consts::get_static(self.cx, def_id)),
                    llextra: ptr::null_mut(),
                    ty: self.monomorphize(&ty),
                }
            }
            mir::Place::Projection(ref projection) => {
                let tr_base = self.const_place(&projection.base, span)?;
                let projected_ty = PlaceTy::Ty { ty: tr_base.ty }
                    .projection_ty(tcx, &projection.elem);
                let base = tr_base.to_const(span);
                let projected_ty = self.monomorphize(&projected_ty).to_ty(tcx);
                let has_metadata = self.cx.type_has_metadata(projected_ty);

                let (projected, llextra) = match projection.elem {
                    mir::ProjectionElem::Deref => {
                        let (base, extra) = if !has_metadata {
                            (base.llval, ptr::null_mut())
                        } else {
                            base.get_fat_ptr(self.cx)
                        };
                        if self.cx.statics.borrow().contains_key(&base) {
                            (Base::Static(base), extra)
                        } else if let ty::TyStr = projected_ty.sty {
                            (Base::Str(base), extra)
                        } else {
                            let v = base;
                            let v = self.cx.const_unsized.borrow().get(&v).map_or(v, |&v| v);
                            let mut val = unsafe { llvm::LLVMGetInitializer(v) };
                            if val.is_null() {
                                span_bug!(span, "dereference of non-constant pointer `{:?}`",
                                          Value(base));
                            }
                            let layout = self.cx.layout_of(projected_ty);
                            if let layout::Abi::Scalar(ref scalar) = layout.abi {
                                let i1_type = Type::i1(self.cx);
                                if scalar.is_bool() && val_ty(val) != i1_type {
                                    unsafe {
                                        val = llvm::LLVMConstTrunc(val, i1_type.to_ref());
                                    }
                                }
                            }
                            (Base::Value(val), extra)
                        }
                    }
                    mir::ProjectionElem::Field(ref field, _) => {
                        let llprojected = base.get_field(self.cx, field.index());
                        let llextra = if !has_metadata {
                            ptr::null_mut()
                        } else {
                            tr_base.llextra
                        };
                        (Base::Value(llprojected), llextra)
                    }
                    mir::ProjectionElem::Index(index) => {
                        let index = &mir::Operand::Copy(mir::Place::Local(index));
                        let llindex = self.const_operand(index, span)?.llval;

                        let iv = if let Some(iv) = common::const_to_opt_u128(llindex, false) {
                            iv
                        } else {
                            span_bug!(span, "index is not an integer-constant expression")
                        };

                        // Produce an undef instead of a LLVM assertion on OOB.
                        let len = common::const_to_uint(tr_base.len(self.cx));
                        let llelem = if iv < len as u128 {
                            const_get_elt(base.llval, iv as u64)
                        } else {
                            C_undef(self.cx.layout_of(projected_ty).llvm_type(self.cx))
                        };

                        (Base::Value(llelem), ptr::null_mut())
                    }
                    _ => span_bug!(span, "{:?} in constant", projection.elem)
                };
                ConstPlace {
                    base: projected,
                    llextra,
                    ty: projected_ty
                }
            }
        };
        Ok(place)
    }

    fn const_operand(&self, operand: &mir::Operand<'tcx>, span: Span)
                     -> Result<Const<'tcx>, ConstEvalErr<'tcx>> {
        debug!("const_operand({:?} @ {:?})", operand, span);
        let result = match *operand {
            mir::Operand::Copy(ref place) |
            mir::Operand::Move(ref place) => {
                Ok(self.const_place(place, span)?.to_const(span))
            }

            mir::Operand::Constant(ref constant) => {
                let ty = self.monomorphize(&constant.ty);
                match constant.literal.clone() {
                    mir::Literal::Promoted { index } => {
                        let mir = &self.mir.promoted[index];
                        MirConstContext::new(self.cx, mir, self.substs, IndexVec::new()).trans()
                    }
                    mir::Literal::Value { value } => {
                        if let ConstVal::Unevaluated(def_id, substs) = value.val {
                            let substs = self.monomorphize(&substs);
                            MirConstContext::trans_def(self.cx, def_id, substs, IndexVec::new())
                        } else {
                            Ok(Const::from_constval(self.cx, &value.val, ty))
                        }
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
        let llunitty = self.cx.layout_of(elem_ty).llvm_type(self.cx);
        // If the array contains enums, an LLVM array won't work.
        let val = if fields.iter().all(|&f| val_ty(f) == llunitty) {
            C_array(llunitty, fields)
        } else {
            C_struct(self.cx, fields, false)
        };
        Const::new(val, array_ty)
    }

    fn const_rvalue(&self, rvalue: &mir::Rvalue<'tcx>,
                    dest_ty: Ty<'tcx>, span: Span)
                    -> Result<Const<'tcx>, ConstEvalErr<'tcx>> {
        let tcx = self.cx.tcx;
        debug!("const_rvalue({:?}: {:?} @ {:?})", rvalue, dest_ty, span);
        let val = match *rvalue {
            mir::Rvalue::Use(ref operand) => self.const_operand(operand, span)?,

            mir::Rvalue::Repeat(ref elem, count) => {
                let elem = self.const_operand(elem, span)?;
                let size = count.as_u64();
                assert_eq!(size as usize as u64, size);
                let fields = vec![elem.llval; size as usize];
                self.const_array(dest_ty, &fields)
            }

            mir::Rvalue::Aggregate(box mir::AggregateKind::Array(_), ref operands) => {
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

                self.const_array(dest_ty, &fields)
            }

            mir::Rvalue::Aggregate(ref kind, ref operands) => {
                // Make sure to evaluate all operands to
                // report as many errors as we possibly can.
                let mut fields = Vec::with_capacity(operands.len());
                let mut failure = Ok(());
                for operand in operands {
                    match self.const_operand(operand, span) {
                        Ok(val) => fields.push(val),
                        Err(err) => if failure.is_ok() { failure = Err(err); }
                    }
                }
                failure?;

                trans_const_adt(self.cx, dest_ty, kind, &fields)
            }

            mir::Rvalue::Cast(ref kind, ref source, cast_ty) => {
                let operand = self.const_operand(source, span)?;
                let cast_ty = self.monomorphize(&cast_ty);

                let val = match *kind {
                    mir::CastKind::ReifyFnPointer => {
                        match operand.ty.sty {
                            ty::TyFnDef(def_id, substs) => {
                                callee::resolve_and_get_fn(self.cx, def_id, substs)
                            }
                            _ => {
                                span_bug!(span, "{} cannot be reified to a fn ptr",
                                          operand.ty)
                            }
                        }
                    }
                    mir::CastKind::ClosureFnPointer => {
                        match operand.ty.sty {
                            ty::TyClosure(def_id, substs) => {
                                // Get the def_id for FnOnce::call_once
                                let fn_once = tcx.lang_items().fn_once_trait().unwrap();
                                let call_once = tcx
                                    .global_tcx().associated_items(fn_once)
                                    .find(|it| it.kind == ty::AssociatedKind::Method)
                                    .unwrap().def_id;
                                // Now create its substs [Closure, Tuple]
                                let input = substs.closure_sig(def_id, tcx).input(0);
                                let input = tcx.erase_late_bound_regions_and_normalize(&input);
                                let substs = tcx.mk_substs([operand.ty, input]
                                    .iter().cloned().map(Kind::from));
                                callee::resolve_and_get_fn(self.cx, call_once, substs)
                            }
                            _ => {
                                bug!("{} cannot be cast to a fn ptr", operand.ty)
                            }
                        }
                    }
                    mir::CastKind::UnsafeFnPointer => {
                        // this is a no-op at the LLVM level
                        operand.llval
                    }
                    mir::CastKind::Unsize => {
                        let pointee_ty = operand.ty.builtin_deref(true, ty::NoPreference)
                            .expect("consts: unsizing got non-pointer type").ty;
                        let (base, old_info) = if !self.cx.type_is_sized(pointee_ty) {
                            // Normally, the source is a thin pointer and we are
                            // adding extra info to make a fat pointer. The exception
                            // is when we are upcasting an existing object fat pointer
                            // to use a different vtable. In that case, we want to
                            // load out the original data pointer so we can repackage
                            // it.
                            let (base, extra) = operand.get_fat_ptr(self.cx);
                            (base, Some(extra))
                        } else {
                            (operand.llval, None)
                        };

                        let unsized_ty = cast_ty.builtin_deref(true, ty::NoPreference)
                            .expect("consts: unsizing got non-pointer target type").ty;
                        let ptr_ty = self.cx.layout_of(unsized_ty).llvm_type(self.cx).ptr_to();
                        let base = consts::ptrcast(base, ptr_ty);
                        let info = base::unsized_info(self.cx, pointee_ty,
                                                      unsized_ty, old_info);

                        if old_info.is_none() {
                            let prev_const = self.cx.const_unsized.borrow_mut()
                                                     .insert(base, operand.llval);
                            assert!(prev_const.is_none() || prev_const == Some(operand.llval));
                        }
                        C_fat_ptr(self.cx, base, info)
                    }
                    mir::CastKind::Misc if self.cx.layout_of(operand.ty).is_llvm_immediate() => {
                        let r_t_in = CastTy::from_ty(operand.ty).expect("bad input type for cast");
                        let r_t_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                        let cast_layout = self.cx.layout_of(cast_ty);
                        assert!(cast_layout.is_llvm_immediate());
                        let ll_t_out = cast_layout.immediate_llvm_type(self.cx);
                        let llval = operand.llval;

                        let mut signed = false;
                        let l = self.cx.layout_of(operand.ty);
                        if let layout::Abi::Scalar(ref scalar) = l.abi {
                            if let layout::Int(_, true) = scalar.value {
                                signed = true;
                            }
                        }

                        unsafe {
                            match (r_t_in, r_t_out) {
                                (CastTy::Int(_), CastTy::Int(_)) => {
                                    let s = signed as llvm::Bool;
                                    llvm::LLVMConstIntCast(llval, ll_t_out.to_ref(), s)
                                }
                                (CastTy::Int(_), CastTy::Float) => {
                                    cast_const_int_to_float(self.cx, llval, signed, ll_t_out)
                                }
                                (CastTy::Float, CastTy::Float) => {
                                    llvm::LLVMConstFPCast(llval, ll_t_out.to_ref())
                                }
                                (CastTy::Float, CastTy::Int(IntTy::I)) => {
                                    cast_const_float_to_int(self.cx, &operand,
                                                            true, ll_t_out, span)
                                }
                                (CastTy::Float, CastTy::Int(_)) => {
                                    cast_const_float_to_int(self.cx, &operand,
                                                            false, ll_t_out, span)
                                }
                                (CastTy::Ptr(_), CastTy::Ptr(_)) |
                                (CastTy::FnPtr, CastTy::Ptr(_)) |
                                (CastTy::RPtr(_), CastTy::Ptr(_)) => {
                                    consts::ptrcast(llval, ll_t_out)
                                }
                                (CastTy::Int(_), CastTy::Ptr(_)) => {
                                    let s = signed as llvm::Bool;
                                    let usize_llval = llvm::LLVMConstIntCast(llval,
                                        self.cx.isize_ty.to_ref(), s);
                                    llvm::LLVMConstIntToPtr(usize_llval, ll_t_out.to_ref())
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
                        let l = self.cx.layout_of(operand.ty);
                        let cast = self.cx.layout_of(cast_ty);
                        if l.is_llvm_scalar_pair() {
                            let (data_ptr, meta) = operand.get_fat_ptr(self.cx);
                            if cast.is_llvm_scalar_pair() {
                                let data_cast = consts::ptrcast(data_ptr,
                                    cast.scalar_pair_element_llvm_type(self.cx, 0));
                                C_fat_ptr(self.cx, data_cast, meta)
                            } else { // cast to thin-ptr
                                // Cast of fat-ptr to thin-ptr is an extraction of data-ptr and
                                // pointer-cast of that pointer to desired pointer type.
                                let llcast_ty = cast.immediate_llvm_type(self.cx);
                                consts::ptrcast(data_ptr, llcast_ty)
                            }
                        } else {
                            bug!("Unexpected non-fat-pointer operand")
                        }
                    }
                };
                Const::new(val, cast_ty)
            }

            mir::Rvalue::Ref(_, bk, ref place) => {
                let tr_place = self.const_place(place, span)?;

                let ty = tr_place.ty;
                let ref_ty = tcx.mk_ref(tcx.types.re_erased,
                    ty::TypeAndMut { ty: ty, mutbl: bk.to_mutbl_lossy() });

                let base = match tr_place.base {
                    Base::Value(llval) => {
                        // FIXME: may be wrong for &*(&simd_vec as &fmt::Debug)
                        let align = if self.cx.type_is_sized(ty) {
                            self.cx.align_of(ty)
                        } else {
                            self.cx.tcx.data_layout.pointer_align
                        };
                        if bk == mir::BorrowKind::Mut {
                            consts::addr_of_mut(self.cx, llval, align, "ref_mut")
                        } else {
                            consts::addr_of(self.cx, llval, align, "ref")
                        }
                    }
                    Base::Str(llval) |
                    Base::Static(llval) => llval
                };

                let ptr = if self.cx.type_is_sized(ty) {
                    base
                } else {
                    C_fat_ptr(self.cx, base, tr_place.llextra)
                };
                Const::new(ptr, ref_ty)
            }

            mir::Rvalue::Len(ref place) => {
                let tr_place = self.const_place(place, span)?;
                Const::new(tr_place.len(self.cx), tcx.types.usize)
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
                let binop_ty = tcx.intern_tup(&[val_ty, tcx.types.bool], false);
                let (lhs, rhs) = (lhs.llval, rhs.llval);
                assert!(!ty.is_fp());

                match const_scalar_checked_binop(tcx, op, lhs, rhs, ty) {
                    Some((llval, of)) => {
                        trans_const_adt(self.cx, binop_ty, &mir::AggregateKind::Tuple, &[
                            Const::new(llval, val_ty),
                            Const::new(C_bool(self.cx, of), tcx.types.bool)
                        ])
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

            mir::Rvalue::NullaryOp(mir::NullOp::SizeOf, ty) => {
                assert!(self.cx.type_is_sized(ty));
                let llval = C_usize(self.cx, self.cx.size_of(ty).bytes());
                Const::new(llval, tcx.types.usize)
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
                                                   tcx.sess.target.isize_ty)),
        ty::TyUint(uint_type) => const_to_opt_u128(value, false)
            .and_then(|input| ConstInt::new_unsigned(input, uint_type,
                                                     tcx.sess.target.usize_ty)),
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
            mir::BinOp::Offset => unreachable!("BinOp::Offset in const-eval!")
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

unsafe fn cast_const_float_to_int(cx: &CodegenCx,
                                  operand: &Const,
                                  signed: bool,
                                  int_ty: Type,
                                  span: Span) -> ValueRef {
    let llval = operand.llval;
    let float_bits = match operand.ty.sty {
        ty::TyFloat(fty) => fty.bit_width(),
        _ => bug!("cast_const_float_to_int: operand not a float"),
    };
    // Note: this breaks if llval is a complex constant expression rather than a simple constant.
    // One way that might happen would be if addresses could be turned into integers in constant
    // expressions, but that doesn't appear to be possible?
    // In any case, an ICE is better than producing undef.
    let llval_bits = consts::bitcast(llval, Type::ix(cx, float_bits as u64));
    let bits = const_to_opt_u128(llval_bits, false).unwrap_or_else(|| {
        panic!("could not get bits of constant float {:?}",
               Value(llval));
    });
    let int_width = int_ty.int_width() as usize;
    // Try to convert, but report an error for overflow and NaN. This matches HIR const eval.
    let cast_result = match float_bits {
        32 if signed => ieee::Single::from_bits(bits).to_i128(int_width).map(|v| v as u128),
        64 if signed => ieee::Double::from_bits(bits).to_i128(int_width).map(|v| v as u128),
        32 => ieee::Single::from_bits(bits).to_u128(int_width),
        64 => ieee::Double::from_bits(bits).to_u128(int_width),
        n => bug!("unsupported float width {}", n),
    };
    if cast_result.status.contains(Status::INVALID_OP) {
        let err = ConstEvalErr { span: span, kind: ErrKind::CannotCast };
        err.report(cx.tcx, span, "expression");
    }
    C_uint_big(int_ty, cast_result.value)
}

unsafe fn cast_const_int_to_float(cx: &CodegenCx,
                                  llval: ValueRef,
                                  signed: bool,
                                  float_ty: Type) -> ValueRef {
    // Note: this breaks if llval is a complex constant expression rather than a simple constant.
    // One way that might happen would be if addresses could be turned into integers in constant
    // expressions, but that doesn't appear to be possible?
    // In any case, an ICE is better than producing undef.
    let value = const_to_opt_u128(llval, signed).unwrap_or_else(|| {
        panic!("could not get z128 value of constant integer {:?}",
               Value(llval));
    });
    if signed {
        llvm::LLVMConstSIToFP(llval, float_ty.to_ref())
    } else if float_ty.float_width() == 32 && value >= MAX_F32_PLUS_HALF_ULP {
        // We're casting to f32 and the value is > f32::MAX + 0.5 ULP -> round up to infinity.
        let infinity_bits = C_u32(cx, ieee::Single::INFINITY.to_bits() as u32);
        consts::bitcast(infinity_bits, float_ty)
    } else {
        llvm::LLVMConstUIToFP(llval, float_ty.to_ref())
    }
}

impl<'a, 'tcx> FunctionCx<'a, 'tcx> {
    pub fn trans_constant(&mut self,
                          bx: &Builder<'a, 'tcx>,
                          constant: &mir::Constant<'tcx>)
                          -> Const<'tcx>
    {
        debug!("trans_constant({:?})", constant);
        let ty = self.monomorphize(&constant.ty);
        let result = match constant.literal.clone() {
            mir::Literal::Promoted { index } => {
                let mir = &self.mir.promoted[index];
                MirConstContext::new(bx.cx, mir, self.param_substs, IndexVec::new()).trans()
            }
            mir::Literal::Value { value } => {
                if let ConstVal::Unevaluated(def_id, substs) = value.val {
                    let substs = self.monomorphize(&substs);
                    MirConstContext::trans_def(bx.cx, def_id, substs, IndexVec::new())
                } else {
                    Ok(Const::from_constval(bx.cx, &value.val, ty))
                }
            }
        };

        let result = result.unwrap_or_else(|_| {
            // We've errored, so we don't have to produce working code.
            let llty = bx.cx.layout_of(ty).llvm_type(bx.cx);
            Const::new(C_undef(llty), ty)
        });

        debug!("trans_constant({:?}) = {:?}", constant, result);
        result
    }
}


pub fn trans_static_initializer<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    def_id: DefId)
    -> Result<ValueRef, ConstEvalErr<'tcx>>
{
    MirConstContext::trans_def(cx, def_id, Substs::empty(), IndexVec::new())
        .map(|c| c.llval)
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
fn trans_const_adt<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    t: Ty<'tcx>,
    kind: &mir::AggregateKind,
    vals: &[Const<'tcx>]
) -> Const<'tcx> {
    let l = cx.layout_of(t);
    let variant_index = match *kind {
        mir::AggregateKind::Adt(_, index, _, _) => index,
        _ => 0,
    };

    if let layout::Abi::Uninhabited = l.abi {
        return Const::new(C_undef(l.llvm_type(cx)), t);
    }

    match l.variants {
        layout::Variants::Single { index } => {
            assert_eq!(variant_index, index);
            if let layout::FieldPlacement::Union(_) = l.fields {
                assert_eq!(variant_index, 0);
                assert_eq!(vals.len(), 1);
                let (field_size, field_align) = cx.size_and_align_of(vals[0].ty);
                let contents = [
                    vals[0].llval,
                    padding(cx, l.size - field_size)
                ];

                let packed = l.align.abi() < field_align.abi();
                Const::new(C_struct(cx, &contents, packed), t)
            } else {
                if let layout::Abi::Vector { .. } = l.abi {
                    if let layout::FieldPlacement::Array { .. } = l.fields {
                        return Const::new(C_vector(&vals.iter().map(|x| x.llval)
                            .collect::<Vec<_>>()), t);
                    }
                }
                build_const_struct(cx, l, vals, None)
            }
        }
        layout::Variants::Tagged { .. } => {
            let discr = match *kind {
                mir::AggregateKind::Adt(adt_def, _, _, _) => {
                    adt_def.discriminant_for_variant(cx.tcx, variant_index)
                           .to_u128_unchecked() as u64
                },
                _ => 0,
            };
            let discr_field = l.field(cx, 0);
            let discr = C_int(discr_field.llvm_type(cx), discr as i64);
            if let layout::Abi::Scalar(_) = l.abi {
                Const::new(discr, t)
            } else {
                let discr = Const::new(discr, discr_field.ty);
                build_const_struct(cx, l.for_variant(cx, variant_index), vals, Some(discr))
            }
        }
        layout::Variants::NicheFilling {
            dataful_variant,
            ref niche_variants,
            niche_start,
            ..
        } => {
            if variant_index == dataful_variant {
                build_const_struct(cx, l.for_variant(cx, dataful_variant), vals, None)
            } else {
                let niche = l.field(cx, 0);
                let niche_llty = niche.llvm_type(cx);
                let niche_value = ((variant_index - niche_variants.start) as u128)
                    .wrapping_add(niche_start);
                // FIXME(eddyb) Check the actual primitive type here.
                let niche_llval = if niche_value == 0 {
                    // HACK(eddyb) Using `C_null` as it works on all types.
                    C_null(niche_llty)
                } else {
                    C_uint_big(niche_llty, niche_value)
                };
                build_const_struct(cx, l, &[Const::new(niche_llval, niche.ty)], None)
            }
        }
    }
}

/// Building structs is a little complicated, because we might need to
/// insert padding if a field's value is less aligned than its type.
///
/// Continuing the example from `trans_const_adt`, a value of type `(u32,
/// E)` should have the `E` at offset 8, but if that field's
/// initializer is 4-byte aligned then simply translating the tuple as
/// a two-element struct will locate it at offset 4, and accesses to it
/// will read the wrong memory.
fn build_const_struct<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                layout: layout::TyLayout<'tcx>,
                                vals: &[Const<'tcx>],
                                discr: Option<Const<'tcx>>)
                                -> Const<'tcx> {
    assert_eq!(vals.len(), layout.fields.count());

    match layout.abi {
        layout::Abi::Scalar(_) |
        layout::Abi::ScalarPair(..) |
        layout::Abi::Vector { .. } if discr.is_none() => {
            let mut non_zst_fields = vals.iter().enumerate().map(|(i, f)| {
                (f, layout.fields.offset(i))
            }).filter(|&(f, _)| !cx.layout_of(f.ty).is_zst());
            match (non_zst_fields.next(), non_zst_fields.next()) {
                (Some((x, offset)), None) if offset.bytes() == 0 => {
                    return Const::new(x.llval, layout.ty);
                }
                (Some((a, a_offset)), Some((b, _))) if a_offset.bytes() == 0 => {
                    return Const::new(C_struct(cx, &[a.llval, b.llval], false), layout.ty);
                }
                (Some((a, _)), Some((b, b_offset))) if b_offset.bytes() == 0 => {
                    return Const::new(C_struct(cx, &[b.llval, a.llval], false), layout.ty);
                }
                _ => {}
            }
        }
        _ => {}
    }

    // offset of current value
    let mut packed = false;
    let mut offset = Size::from_bytes(0);
    let mut cfields = Vec::new();
    cfields.reserve(discr.is_some() as usize + 1 + layout.fields.count() * 2);

    if let Some(discr) = discr {
        let (field_size, field_align) = cx.size_and_align_of(discr.ty);
        packed |= layout.align.abi() < field_align.abi();
        cfields.push(discr.llval);
        offset = field_size;
    }

    let parts = layout.fields.index_by_increasing_offset().map(|i| {
        (vals[i], layout.fields.offset(i))
    });
    for (val, target_offset) in parts {
        let (field_size, field_align) = cx.size_and_align_of(val.ty);
        packed |= layout.align.abi() < field_align.abi();
        cfields.push(padding(cx, target_offset - offset));
        cfields.push(val.llval);
        offset = target_offset + field_size;
    }

    // Pad to the size of the whole type, not e.g. the variant.
    cfields.push(padding(cx, cx.size_of(layout.ty) - offset));

    Const::new(C_struct(cx, &cfields, packed), layout.ty)
}

fn padding(cx: &CodegenCx, size: Size) -> ValueRef {
    C_undef(Type::array(&Type::i8(cx), size.bytes()))
}
