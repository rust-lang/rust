// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{self, ValueRef, AttributePlace};
use base;
use builder::{Builder, MemFlags};
use common::{ty_fn_sig, C_usize};
use context::CodegenCx;
use mir::place::PlaceRef;
use mir::operand::OperandValue;
use type_::Type;
use type_of::{LayoutLlvmExt, PointerKind};

use rustc_target::abi::{LayoutOf, Size, TyLayout};
use rustc::ty::{self, Ty};
use rustc::ty::layout;

use libc::c_uint;

pub use rustc_target::spec::abi::Abi;
pub use rustc::ty::layout::{FAT_PTR_ADDR, FAT_PTR_EXTRA};
pub use rustc_target::abi::call::*;

macro_rules! for_each_kind {
    ($flags: ident, $f: ident, $($kind: ident),+) => ({
        $(if $flags.contains(ArgAttribute::$kind) { $f(llvm::Attribute::$kind) })+
    })
}

trait ArgAttributeExt {
    fn for_each_kind<F>(&self, f: F) where F: FnMut(llvm::Attribute);
}

impl ArgAttributeExt for ArgAttribute {
    fn for_each_kind<F>(&self, mut f: F) where F: FnMut(llvm::Attribute) {
        for_each_kind!(self, f,
                       ByVal, NoAlias, NoCapture, NonNull, ReadOnly, SExt, StructRet, ZExt, InReg)
    }
}

pub trait ArgAttributesExt {
    fn apply_llfn(&self, idx: AttributePlace, llfn: ValueRef);
    fn apply_callsite(&self, idx: AttributePlace, callsite: ValueRef);
}

impl ArgAttributesExt for ArgAttributes {
    fn apply_llfn(&self, idx: AttributePlace, llfn: ValueRef) {
        let mut regular = self.regular;
        unsafe {
            let deref = self.pointee_size.bytes();
            if deref != 0 {
                if regular.contains(ArgAttribute::NonNull) {
                    llvm::LLVMRustAddDereferenceableAttr(llfn,
                                                         idx.as_uint(),
                                                         deref);
                } else {
                    llvm::LLVMRustAddDereferenceableOrNullAttr(llfn,
                                                               idx.as_uint(),
                                                               deref);
                }
                regular -= ArgAttribute::NonNull;
            }
            if let Some(align) = self.pointee_align {
                llvm::LLVMRustAddAlignmentAttr(llfn,
                                               idx.as_uint(),
                                               align.abi() as u32);
            }
            regular.for_each_kind(|attr| attr.apply_llfn(idx, llfn));
        }
    }

    fn apply_callsite(&self, idx: AttributePlace, callsite: ValueRef) {
        let mut regular = self.regular;
        unsafe {
            let deref = self.pointee_size.bytes();
            if deref != 0 {
                if regular.contains(ArgAttribute::NonNull) {
                    llvm::LLVMRustAddDereferenceableCallSiteAttr(callsite,
                                                                 idx.as_uint(),
                                                                 deref);
                } else {
                    llvm::LLVMRustAddDereferenceableOrNullCallSiteAttr(callsite,
                                                                       idx.as_uint(),
                                                                       deref);
                }
                regular -= ArgAttribute::NonNull;
            }
            if let Some(align) = self.pointee_align {
                llvm::LLVMRustAddAlignmentCallSiteAttr(callsite,
                                                       idx.as_uint(),
                                                       align.abi() as u32);
            }
            regular.for_each_kind(|attr| attr.apply_callsite(idx, callsite));
        }
    }
}

pub trait LlvmType {
    fn llvm_type(&self, cx: &CodegenCx) -> Type;
}

impl LlvmType for Reg {
    fn llvm_type(&self, cx: &CodegenCx) -> Type {
        match self.kind {
            RegKind::Integer => Type::ix(cx, self.size.bits()),
            RegKind::Float => {
                match self.size.bits() {
                    32 => Type::f32(cx),
                    64 => Type::f64(cx),
                    _ => bug!("unsupported float: {:?}", self)
                }
            }
            RegKind::Vector => {
                Type::vector(&Type::i8(cx), self.size.bytes())
            }
        }
    }
}

impl LlvmType for CastTarget {
    fn llvm_type(&self, cx: &CodegenCx) -> Type {
        let rest_ll_unit = self.rest.unit.llvm_type(cx);
        let (rest_count, rem_bytes) = if self.rest.unit.size.bytes() == 0 {
            (0, 0)
        } else {
            (self.rest.total.bytes() / self.rest.unit.size.bytes(),
            self.rest.total.bytes() % self.rest.unit.size.bytes())
        };

        if self.prefix.iter().all(|x| x.is_none()) {
            // Simplify to a single unit when there is no prefix and size <= unit size
            if self.rest.total <= self.rest.unit.size {
                return rest_ll_unit;
            }

            // Simplify to array when all chunks are the same size and type
            if rem_bytes == 0 {
                return Type::array(&rest_ll_unit, rest_count);
            }
        }

        // Create list of fields in the main structure
        let mut args: Vec<_> =
            self.prefix.iter().flat_map(|option_kind| option_kind.map(
                    |kind| Reg { kind: kind, size: self.prefix_chunk }.llvm_type(cx)))
            .chain((0..rest_count).map(|_| rest_ll_unit))
            .collect();

        // Append final integer
        if rem_bytes != 0 {
            // Only integers can be really split further.
            assert_eq!(self.rest.unit.kind, RegKind::Integer);
            args.push(Type::ix(cx, rem_bytes * 8));
        }

        Type::struct_(cx, &args, false)
    }
}

pub trait ArgTypeExt<'a, 'tcx> {
    fn memory_ty(&self, cx: &CodegenCx<'a, 'tcx>) -> Type;
    fn store(&self, bx: &Builder<'a, 'tcx>, val: ValueRef, dst: PlaceRef<'tcx>);
    fn store_fn_arg(&self, bx: &Builder<'a, 'tcx>, idx: &mut usize, dst: PlaceRef<'tcx>);
}

impl<'a, 'tcx> ArgTypeExt<'a, 'tcx> for ArgType<'tcx, Ty<'tcx>> {
    /// Get the LLVM type for a place of the original Rust type of
    /// this argument/return, i.e. the result of `type_of::type_of`.
    fn memory_ty(&self, cx: &CodegenCx<'a, 'tcx>) -> Type {
        self.layout.llvm_type(cx)
    }

    /// Store a direct/indirect value described by this ArgType into a
    /// place for the original Rust type of this argument/return.
    /// Can be used for both storing formal arguments into Rust variables
    /// or results of call/invoke instructions into their destinations.
    fn store(&self, bx: &Builder<'a, 'tcx>, val: ValueRef, dst: PlaceRef<'tcx>) {
        if self.is_ignore() {
            return;
        }
        let cx = bx.cx;
        if self.is_indirect() {
            OperandValue::Ref(val, self.layout.align).store(bx, dst)
        } else if let PassMode::Cast(cast) = self.mode {
            // FIXME(eddyb): Figure out when the simpler Store is safe, clang
            // uses it for i16 -> {i8, i8}, but not for i24 -> {i8, i8, i8}.
            let can_store_through_cast_ptr = false;
            if can_store_through_cast_ptr {
                let cast_dst = bx.pointercast(dst.llval, cast.llvm_type(cx).ptr_to());
                bx.store(val, cast_dst, self.layout.align);
            } else {
                // The actual return type is a struct, but the ABI
                // adaptation code has cast it into some scalar type.  The
                // code that follows is the only reliable way I have
                // found to do a transform like i64 -> {i32,i32}.
                // Basically we dump the data onto the stack then memcpy it.
                //
                // Other approaches I tried:
                // - Casting rust ret pointer to the foreign type and using Store
                //   is (a) unsafe if size of foreign type > size of rust type and
                //   (b) runs afoul of strict aliasing rules, yielding invalid
                //   assembly under -O (specifically, the store gets removed).
                // - Truncating foreign type to correct integral type and then
                //   bitcasting to the struct type yields invalid cast errors.

                // We instead thus allocate some scratch space...
                let scratch_size = cast.size(cx);
                let scratch_align = cast.align(cx);
                let llscratch = bx.alloca(cast.llvm_type(cx), "abi_cast", scratch_align);
                bx.lifetime_start(llscratch, scratch_size);

                // ...where we first store the value...
                bx.store(val, llscratch, scratch_align);

                // ...and then memcpy it to the intended destination.
                base::call_memcpy(bx,
                                  bx.pointercast(dst.llval, Type::i8p(cx)),
                                  bx.pointercast(llscratch, Type::i8p(cx)),
                                  C_usize(cx, self.layout.size.bytes()),
                                  self.layout.align.min(scratch_align),
                                  MemFlags::empty());

                bx.lifetime_end(llscratch, scratch_size);
            }
        } else {
            OperandValue::Immediate(val).store(bx, dst);
        }
    }

    fn store_fn_arg(&self, bx: &Builder<'a, 'tcx>, idx: &mut usize, dst: PlaceRef<'tcx>) {
        let mut next = || {
            let val = llvm::get_param(bx.llfn(), *idx as c_uint);
            *idx += 1;
            val
        };
        match self.mode {
            PassMode::Ignore => {},
            PassMode::Pair(..) => {
                OperandValue::Pair(next(), next()).store(bx, dst);
            }
            PassMode::Direct(_) | PassMode::Indirect(_) | PassMode::Cast(_) => {
                self.store(bx, next(), dst);
            }
        }
    }
}

pub trait FnTypeExt<'a, 'tcx> {
    fn of_instance(cx: &CodegenCx<'a, 'tcx>, instance: &ty::Instance<'tcx>)
                   -> Self;
    fn new(cx: &CodegenCx<'a, 'tcx>,
           sig: ty::FnSig<'tcx>,
           extra_args: &[Ty<'tcx>]) -> Self;
    fn new_vtable(cx: &CodegenCx<'a, 'tcx>,
                  sig: ty::FnSig<'tcx>,
                  extra_args: &[Ty<'tcx>]) -> Self;
    fn unadjusted(cx: &CodegenCx<'a, 'tcx>,
                  sig: ty::FnSig<'tcx>,
                  extra_args: &[Ty<'tcx>]) -> Self;
    fn adjust_for_abi(&mut self,
                      cx: &CodegenCx<'a, 'tcx>,
                      abi: Abi);
    fn llvm_type(&self, cx: &CodegenCx<'a, 'tcx>) -> Type;
    fn llvm_cconv(&self) -> llvm::CallConv;
    fn apply_attrs_llfn(&self, llfn: ValueRef);
    fn apply_attrs_callsite(&self, bx: &Builder<'a, 'tcx>, callsite: ValueRef);
}

impl<'a, 'tcx> FnTypeExt<'a, 'tcx> for FnType<'tcx, Ty<'tcx>> {
    fn of_instance(cx: &CodegenCx<'a, 'tcx>, instance: &ty::Instance<'tcx>)
                       -> Self {
        let fn_ty = instance.ty(cx.tcx);
        let sig = ty_fn_sig(cx, fn_ty);
        let sig = cx.tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);
        FnType::new(cx, sig, &[])
    }

    fn new(cx: &CodegenCx<'a, 'tcx>,
               sig: ty::FnSig<'tcx>,
               extra_args: &[Ty<'tcx>]) -> Self {
        let mut fn_ty = FnType::unadjusted(cx, sig, extra_args);
        fn_ty.adjust_for_abi(cx, sig.abi);
        fn_ty
    }

    fn new_vtable(cx: &CodegenCx<'a, 'tcx>,
                      sig: ty::FnSig<'tcx>,
                      extra_args: &[Ty<'tcx>]) -> Self {
        let mut fn_ty = FnType::unadjusted(cx, sig, extra_args);
        // Don't pass the vtable, it's not an argument of the virtual fn.
        {
            let self_arg = &mut fn_ty.args[0];
            match self_arg.mode {
                PassMode::Pair(data_ptr, _) => {
                    self_arg.mode = PassMode::Direct(data_ptr);
                }
                _ => bug!("FnType::new_vtable: non-pair self {:?}", self_arg)
            }

            let pointee = self_arg.layout.ty.builtin_deref(true)
                .unwrap_or_else(|| {
                    bug!("FnType::new_vtable: non-pointer self {:?}", self_arg)
                }).ty;
            let fat_ptr_ty = cx.tcx.mk_mut_ptr(pointee);
            self_arg.layout = cx.layout_of(fat_ptr_ty).field(cx, 0);
        }
        fn_ty.adjust_for_abi(cx, sig.abi);
        fn_ty
    }

    fn unadjusted(cx: &CodegenCx<'a, 'tcx>,
                      sig: ty::FnSig<'tcx>,
                      extra_args: &[Ty<'tcx>]) -> Self {
        debug!("FnType::unadjusted({:?}, {:?})", sig, extra_args);

        use self::Abi::*;
        let conv = match cx.sess().target.target.adjust_abi(sig.abi) {
            RustIntrinsic | PlatformIntrinsic |
            Rust | RustCall => Conv::C,

            // It's the ABI's job to select this, not us.
            System => bug!("system abi should be selected elsewhere"),

            Stdcall => Conv::X86Stdcall,
            Fastcall => Conv::X86Fastcall,
            Vectorcall => Conv::X86VectorCall,
            Thiscall => Conv::X86ThisCall,
            C => Conv::C,
            Unadjusted => Conv::C,
            Win64 => Conv::X86_64Win64,
            SysV64 => Conv::X86_64SysV,
            Aapcs => Conv::ArmAapcs,
            PtxKernel => Conv::PtxKernel,
            Msp430Interrupt => Conv::Msp430Intr,
            X86Interrupt => Conv::X86Intr,

            // These API constants ought to be more specific...
            Cdecl => Conv::C,
        };

        let mut inputs = sig.inputs();
        let extra_args = if sig.abi == RustCall {
            assert!(!sig.variadic && extra_args.is_empty());

            match sig.inputs().last().unwrap().sty {
                ty::TyTuple(ref tupled_arguments) => {
                    inputs = &sig.inputs()[0..sig.inputs().len() - 1];
                    tupled_arguments
                }
                _ => {
                    bug!("argument to function with \"rust-call\" ABI \
                          is not a tuple");
                }
            }
        } else {
            assert!(sig.variadic || extra_args.is_empty());
            extra_args
        };

        let target = &cx.sess().target.target;
        let win_x64_gnu = target.target_os == "windows"
                       && target.arch == "x86_64"
                       && target.target_env == "gnu";
        let linux_s390x = target.target_os == "linux"
                       && target.arch == "s390x"
                       && target.target_env == "gnu";
        let rust_abi = match sig.abi {
            RustIntrinsic | PlatformIntrinsic | Rust | RustCall => true,
            _ => false
        };

        // Handle safe Rust thin and fat pointers.
        let adjust_for_rust_scalar = |attrs: &mut ArgAttributes,
                                      scalar: &layout::Scalar,
                                      layout: TyLayout<'tcx, Ty<'tcx>>,
                                      offset: Size,
                                      is_return: bool| {
            // Booleans are always an i1 that needs to be zero-extended.
            if scalar.is_bool() {
                attrs.set(ArgAttribute::ZExt);
                return;
            }

            // Only pointer types handled below.
            if scalar.value != layout::Pointer {
                return;
            }

            if scalar.valid_range.start() < scalar.valid_range.end() {
                if *scalar.valid_range.start() > 0 {
                    attrs.set(ArgAttribute::NonNull);
                }
            }

            if let Some(pointee) = layout.pointee_info_at(cx, offset) {
                if let Some(kind) = pointee.safe {
                    attrs.pointee_size = pointee.size;
                    attrs.pointee_align = Some(pointee.align);

                    // HACK(eddyb) LLVM inserts `llvm.assume` calls when inlining functions
                    // with align attributes, and those calls later block optimizations.
                    if !is_return && !cx.tcx.sess.opts.debugging_opts.arg_align_attributes {
                        attrs.pointee_align = None;
                    }

                    // `Box` pointer parameters never alias because ownership is transferred
                    // `&mut` pointer parameters never alias other parameters,
                    // or mutable global data
                    //
                    // `&T` where `T` contains no `UnsafeCell<U>` is immutable,
                    // and can be marked as both `readonly` and `noalias`, as
                    // LLVM's definition of `noalias` is based solely on memory
                    // dependencies rather than pointer equality
                    let no_alias = match kind {
                        PointerKind::Shared => false,
                        PointerKind::UniqueOwned => true,
                        PointerKind::Frozen |
                        PointerKind::UniqueBorrowed => !is_return
                    };
                    if no_alias {
                        attrs.set(ArgAttribute::NoAlias);
                    }

                    if kind == PointerKind::Frozen && !is_return {
                        attrs.set(ArgAttribute::ReadOnly);
                    }
                }
            }
        };

        let arg_of = |ty: Ty<'tcx>, is_return: bool| {
            let mut arg = ArgType::new(cx.layout_of(ty));
            if arg.layout.is_zst() {
                // For some forsaken reason, x86_64-pc-windows-gnu
                // doesn't ignore zero-sized struct arguments.
                // The same is true for s390x-unknown-linux-gnu.
                if is_return || rust_abi || (!win_x64_gnu && !linux_s390x) {
                    arg.mode = PassMode::Ignore;
                }
            }

            // FIXME(eddyb) other ABIs don't have logic for scalar pairs.
            if !is_return && rust_abi {
                if let layout::Abi::ScalarPair(ref a, ref b) = arg.layout.abi {
                    let mut a_attrs = ArgAttributes::new();
                    let mut b_attrs = ArgAttributes::new();
                    adjust_for_rust_scalar(&mut a_attrs,
                                           a,
                                           arg.layout,
                                           Size::ZERO,
                                           false);
                    adjust_for_rust_scalar(&mut b_attrs,
                                           b,
                                           arg.layout,
                                           a.value.size(cx).abi_align(b.value.align(cx)),
                                           false);
                    arg.mode = PassMode::Pair(a_attrs, b_attrs);
                    return arg;
                }
            }

            if let layout::Abi::Scalar(ref scalar) = arg.layout.abi {
                if let PassMode::Direct(ref mut attrs) = arg.mode {
                    adjust_for_rust_scalar(attrs,
                                           scalar,
                                           arg.layout,
                                           Size::ZERO,
                                           is_return);
                }
            }

            arg
        };

        FnType {
            ret: arg_of(sig.output(), true),
            args: inputs.iter().chain(extra_args.iter()).map(|ty| {
                arg_of(ty, false)
            }).collect(),
            variadic: sig.variadic,
            conv,
        }
    }

    fn adjust_for_abi(&mut self,
                      cx: &CodegenCx<'a, 'tcx>,
                      abi: Abi) {
        if abi == Abi::Unadjusted { return }

        if abi == Abi::Rust || abi == Abi::RustCall ||
           abi == Abi::RustIntrinsic || abi == Abi::PlatformIntrinsic {
            let fixup = |arg: &mut ArgType<'tcx, Ty<'tcx>>| {
                if arg.is_ignore() { return; }

                match arg.layout.abi {
                    layout::Abi::Aggregate { .. } => {}

                    // This is a fun case! The gist of what this is doing is
                    // that we want callers and callees to always agree on the
                    // ABI of how they pass SIMD arguments. If we were to *not*
                    // make these arguments indirect then they'd be immediates
                    // in LLVM, which means that they'd used whatever the
                    // appropriate ABI is for the callee and the caller. That
                    // means, for example, if the caller doesn't have AVX
                    // enabled but the callee does, then passing an AVX argument
                    // across this boundary would cause corrupt data to show up.
                    //
                    // This problem is fixed by unconditionally passing SIMD
                    // arguments through memory between callers and callees
                    // which should get them all to agree on ABI regardless of
                    // target feature sets. Some more information about this
                    // issue can be found in #44367.
                    //
                    // Note that the platform intrinsic ABI is exempt here as
                    // that's how we connect up to LLVM and it's unstable
                    // anyway, we control all calls to it in libstd.
                    layout::Abi::Vector { .. } if abi != Abi::PlatformIntrinsic => {
                        arg.make_indirect();
                        return
                    }

                    _ => return
                }

                let size = arg.layout.size;
                if size > layout::Pointer.size(cx) {
                    arg.make_indirect();
                } else {
                    // We want to pass small aggregates as immediates, but using
                    // a LLVM aggregate type for this leads to bad optimizations,
                    // so we pick an appropriately sized integer type instead.
                    arg.cast_to(Reg {
                        kind: RegKind::Integer,
                        size
                    });
                }
            };
            fixup(&mut self.ret);
            for arg in &mut self.args {
                fixup(arg);
            }
            if let PassMode::Indirect(ref mut attrs) = self.ret.mode {
                attrs.set(ArgAttribute::StructRet);
            }
            return;
        }

        if let Err(msg) = self.adjust_for_cabi(cx, abi) {
            cx.sess().fatal(&msg);
        }
    }

    fn llvm_type(&self, cx: &CodegenCx<'a, 'tcx>) -> Type {
        let mut llargument_tys = Vec::new();

        let llreturn_ty = match self.ret.mode {
            PassMode::Ignore => Type::void(cx),
            PassMode::Direct(_) | PassMode::Pair(..) => {
                self.ret.layout.immediate_llvm_type(cx)
            }
            PassMode::Cast(cast) => cast.llvm_type(cx),
            PassMode::Indirect(_) => {
                llargument_tys.push(self.ret.memory_ty(cx).ptr_to());
                Type::void(cx)
            }
        };

        for arg in &self.args {
            // add padding
            if let Some(ty) = arg.pad {
                llargument_tys.push(ty.llvm_type(cx));
            }

            let llarg_ty = match arg.mode {
                PassMode::Ignore => continue,
                PassMode::Direct(_) => arg.layout.immediate_llvm_type(cx),
                PassMode::Pair(..) => {
                    llargument_tys.push(arg.layout.scalar_pair_element_llvm_type(cx, 0));
                    llargument_tys.push(arg.layout.scalar_pair_element_llvm_type(cx, 1));
                    continue;
                }
                PassMode::Cast(cast) => cast.llvm_type(cx),
                PassMode::Indirect(_) => arg.memory_ty(cx).ptr_to(),
            };
            llargument_tys.push(llarg_ty);
        }

        if self.variadic {
            Type::variadic_func(&llargument_tys, &llreturn_ty)
        } else {
            Type::func(&llargument_tys, &llreturn_ty)
        }
    }

    fn llvm_cconv(&self) -> llvm::CallConv {
        match self.conv {
            Conv::C => llvm::CCallConv,
            Conv::ArmAapcs => llvm::ArmAapcsCallConv,
            Conv::Msp430Intr => llvm::Msp430Intr,
            Conv::PtxKernel => llvm::PtxKernel,
            Conv::X86Fastcall => llvm::X86FastcallCallConv,
            Conv::X86Intr => llvm::X86_Intr,
            Conv::X86Stdcall => llvm::X86StdcallCallConv,
            Conv::X86ThisCall => llvm::X86_ThisCall,
            Conv::X86VectorCall => llvm::X86_VectorCall,
            Conv::X86_64SysV => llvm::X86_64_SysV,
            Conv::X86_64Win64 => llvm::X86_64_Win64,
        }
    }

    fn apply_attrs_llfn(&self, llfn: ValueRef) {
        let mut i = 0;
        let mut apply = |attrs: &ArgAttributes| {
            attrs.apply_llfn(llvm::AttributePlace::Argument(i), llfn);
            i += 1;
        };
        match self.ret.mode {
            PassMode::Direct(ref attrs) => {
                attrs.apply_llfn(llvm::AttributePlace::ReturnValue, llfn);
            }
            PassMode::Indirect(ref attrs) => apply(attrs),
            _ => {}
        }
        for arg in &self.args {
            if arg.pad.is_some() {
                apply(&ArgAttributes::new());
            }
            match arg.mode {
                PassMode::Ignore => {}
                PassMode::Direct(ref attrs) |
                PassMode::Indirect(ref attrs) => apply(attrs),
                PassMode::Pair(ref a, ref b) => {
                    apply(a);
                    apply(b);
                }
                PassMode::Cast(_) => apply(&ArgAttributes::new()),
            }
        }
    }

    fn apply_attrs_callsite(&self, bx: &Builder<'a, 'tcx>, callsite: ValueRef) {
        let mut i = 0;
        let mut apply = |attrs: &ArgAttributes| {
            attrs.apply_callsite(llvm::AttributePlace::Argument(i), callsite);
            i += 1;
        };
        match self.ret.mode {
            PassMode::Direct(ref attrs) => {
                attrs.apply_callsite(llvm::AttributePlace::ReturnValue, callsite);
            }
            PassMode::Indirect(ref attrs) => apply(attrs),
            _ => {}
        }
        if let layout::Abi::Scalar(ref scalar) = self.ret.layout.abi {
            // If the value is a boolean, the range is 0..2 and that ultimately
            // become 0..0 when the type becomes i1, which would be rejected
            // by the LLVM verifier.
            match scalar.value {
                layout::Int(..) if !scalar.is_bool() => {
                    let range = scalar.valid_range_exclusive(bx.cx);
                    if range.start != range.end {
                        // FIXME(nox): This causes very weird type errors about
                        // SHL operators in constants in stage 2 with LLVM 3.9.
                        if unsafe { llvm::LLVMRustVersionMajor() >= 4 } {
                            bx.range_metadata(callsite, range);
                        }
                    }
                }
                _ => {}
            }
        }
        for arg in &self.args {
            if arg.pad.is_some() {
                apply(&ArgAttributes::new());
            }
            match arg.mode {
                PassMode::Ignore => {}
                PassMode::Direct(ref attrs) |
                PassMode::Indirect(ref attrs) => apply(attrs),
                PassMode::Pair(ref a, ref b) => {
                    apply(a);
                    apply(b);
                }
                PassMode::Cast(_) => apply(&ArgAttributes::new()),
            }
        }

        let cconv = self.llvm_cconv();
        if cconv != llvm::CCallConv {
            llvm::SetInstructionCallConv(callsite, cconv);
        }
    }
}
