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
use builder::Builder;
use common::{ty_fn_sig, C_usize};
use context::CodegenCx;
use cabi_x86;
use cabi_x86_64;
use cabi_x86_win64;
use cabi_arm;
use cabi_aarch64;
use cabi_powerpc;
use cabi_powerpc64;
use cabi_s390x;
use cabi_mips;
use cabi_mips64;
use cabi_asmjs;
use cabi_msp430;
use cabi_sparc;
use cabi_sparc64;
use cabi_nvptx;
use cabi_nvptx64;
use cabi_hexagon;
use mir::place::PlaceRef;
use mir::operand::OperandValue;
use type_::Type;
use type_of::{LayoutLlvmExt, PointerKind};

use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, Align, Size, TyLayout};
use rustc::ty::layout::{HasDataLayout, LayoutOf};

use libc::c_uint;
use std::{cmp, iter};

pub use syntax::abi::Abi;
pub use rustc::ty::layout::{FAT_PTR_ADDR, FAT_PTR_EXTRA};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PassMode {
    /// Ignore the argument (useful for empty struct).
    Ignore,
    /// Pass the argument directly.
    Direct(ArgAttributes),
    /// Pass a pair's elements directly in two arguments.
    Pair(ArgAttributes, ArgAttributes),
    /// Pass the argument after casting it, to either
    /// a single uniform or a pair of registers.
    Cast(CastTarget),
    /// Pass the argument indirectly via a hidden pointer.
    Indirect(ArgAttributes),
}

// Hack to disable non_upper_case_globals only for the bitflags! and not for the rest
// of this module
pub use self::attr_impl::ArgAttribute;

#[allow(non_upper_case_globals)]
#[allow(unused)]
mod attr_impl {
    // The subset of llvm::Attribute needed for arguments, packed into a bitfield.
    bitflags! {
        #[derive(Default)]
        pub struct ArgAttribute: u16 {
            const ByVal     = 1 << 0;
            const NoAlias   = 1 << 1;
            const NoCapture = 1 << 2;
            const NonNull   = 1 << 3;
            const ReadOnly  = 1 << 4;
            const SExt      = 1 << 5;
            const StructRet = 1 << 6;
            const ZExt      = 1 << 7;
            const InReg     = 1 << 8;
        }
    }
}

macro_rules! for_each_kind {
    ($flags: ident, $f: ident, $($kind: ident),+) => ({
        $(if $flags.contains(ArgAttribute::$kind) { $f(llvm::Attribute::$kind) })+
    })
}

impl ArgAttribute {
    fn for_each_kind<F>(&self, mut f: F) where F: FnMut(llvm::Attribute) {
        for_each_kind!(self, f,
                       ByVal, NoAlias, NoCapture, NonNull, ReadOnly, SExt, StructRet, ZExt, InReg)
    }
}

/// A compact representation of LLVM attributes (at least those relevant for this module)
/// that can be manipulated without interacting with LLVM's Attribute machinery.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct ArgAttributes {
    regular: ArgAttribute,
    pointee_size: Size,
    pointee_align: Option<Align>
}

impl ArgAttributes {
    fn new() -> Self {
        ArgAttributes {
            regular: ArgAttribute::default(),
            pointee_size: Size::from_bytes(0),
            pointee_align: None,
        }
    }

    pub fn set(&mut self, attr: ArgAttribute) -> &mut Self {
        self.regular = self.regular | attr;
        self
    }

    pub fn contains(&self, attr: ArgAttribute) -> bool {
        self.regular.contains(attr)
    }

    pub fn apply_llfn(&self, idx: AttributePlace, llfn: ValueRef) {
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

    pub fn apply_callsite(&self, idx: AttributePlace, callsite: ValueRef) {
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
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RegKind {
    Integer,
    Float,
    Vector
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Reg {
    pub kind: RegKind,
    pub size: Size,
}

macro_rules! reg_ctor {
    ($name:ident, $kind:ident, $bits:expr) => {
        pub fn $name() -> Reg {
            Reg {
                kind: RegKind::$kind,
                size: Size::from_bits($bits)
            }
        }
    }
}

impl Reg {
    reg_ctor!(i8, Integer, 8);
    reg_ctor!(i16, Integer, 16);
    reg_ctor!(i32, Integer, 32);
    reg_ctor!(i64, Integer, 64);

    reg_ctor!(f32, Float, 32);
    reg_ctor!(f64, Float, 64);
}

impl Reg {
    pub fn align(&self, cx: &CodegenCx) -> Align {
        let dl = cx.data_layout();
        match self.kind {
            RegKind::Integer => {
                match self.size.bits() {
                    1 => dl.i1_align,
                    2...8 => dl.i8_align,
                    9...16 => dl.i16_align,
                    17...32 => dl.i32_align,
                    33...64 => dl.i64_align,
                    65...128 => dl.i128_align,
                    _ => bug!("unsupported integer: {:?}", self)
                }
            }
            RegKind::Float => {
                match self.size.bits() {
                    32 => dl.f32_align,
                    64 => dl.f64_align,
                    _ => bug!("unsupported float: {:?}", self)
                }
            }
            RegKind::Vector => dl.vector_align(self.size)
        }
    }

    pub fn llvm_type(&self, cx: &CodegenCx) -> Type {
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

/// An argument passed entirely registers with the
/// same kind (e.g. HFA / HVA on PPC64 and AArch64).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Uniform {
    pub unit: Reg,

    /// The total size of the argument, which can be:
    /// * equal to `unit.size` (one scalar/vector)
    /// * a multiple of `unit.size` (an array of scalar/vectors)
    /// * if `unit.kind` is `Integer`, the last element
    ///   can be shorter, i.e. `{ i64, i64, i32 }` for
    ///   64-bit integers with a total size of 20 bytes
    pub total: Size,
}

impl From<Reg> for Uniform {
    fn from(unit: Reg) -> Uniform {
        Uniform {
            unit,
            total: unit.size
        }
    }
}

impl Uniform {
    pub fn align(&self, cx: &CodegenCx) -> Align {
        self.unit.align(cx)
    }

    pub fn llvm_type(&self, cx: &CodegenCx) -> Type {
        let llunit = self.unit.llvm_type(cx);

        if self.total <= self.unit.size {
            return llunit;
        }

        let count = self.total.bytes() / self.unit.size.bytes();
        let rem_bytes = self.total.bytes() % self.unit.size.bytes();

        if rem_bytes == 0 {
            return Type::array(&llunit, count);
        }

        // Only integers can be really split further.
        assert_eq!(self.unit.kind, RegKind::Integer);

        let args: Vec<_> = (0..count).map(|_| llunit)
            .chain(iter::once(Type::ix(cx, rem_bytes * 8)))
            .collect();

        Type::struct_(cx, &args, false)
    }
}

pub trait LayoutExt<'tcx> {
    fn is_aggregate(&self) -> bool;
    fn homogeneous_aggregate<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> Option<Reg>;
}

impl<'tcx> LayoutExt<'tcx> for TyLayout<'tcx> {
    fn is_aggregate(&self) -> bool {
        match self.abi {
            layout::Abi::Uninhabited |
            layout::Abi::Scalar(_) |
            layout::Abi::Vector { .. } => false,
            layout::Abi::ScalarPair(..) |
            layout::Abi::Aggregate { .. } => true
        }
    }

    fn homogeneous_aggregate<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> Option<Reg> {
        match self.abi {
            layout::Abi::Uninhabited => None,

            // The primitive for this algorithm.
            layout::Abi::Scalar(ref scalar) => {
                let kind = match scalar.value {
                    layout::Int(..) |
                    layout::Pointer => RegKind::Integer,
                    layout::F32 |
                    layout::F64 => RegKind::Float
                };
                Some(Reg {
                    kind,
                    size: self.size
                })
            }

            layout::Abi::Vector { .. } => {
                Some(Reg {
                    kind: RegKind::Vector,
                    size: self.size
                })
            }

            layout::Abi::ScalarPair(..) |
            layout::Abi::Aggregate { .. } => {
                let mut total = Size::from_bytes(0);
                let mut result = None;

                let is_union = match self.fields {
                    layout::FieldPlacement::Array { count, .. } => {
                        if count > 0 {
                            return self.field(cx, 0).homogeneous_aggregate(cx);
                        } else {
                            return None;
                        }
                    }
                    layout::FieldPlacement::Union(_) => true,
                    layout::FieldPlacement::Arbitrary { .. } => false
                };

                for i in 0..self.fields.count() {
                    if !is_union && total != self.fields.offset(i) {
                        return None;
                    }

                    let field = self.field(cx, i);
                    match (result, field.homogeneous_aggregate(cx)) {
                        // The field itself must be a homogeneous aggregate.
                        (_, None) => return None,
                        // If this is the first field, record the unit.
                        (None, Some(unit)) => {
                            result = Some(unit);
                        }
                        // For all following fields, the unit must be the same.
                        (Some(prev_unit), Some(unit)) => {
                            if prev_unit != unit {
                                return None;
                            }
                        }
                    }

                    // Keep track of the offset (without padding).
                    let size = field.size;
                    if is_union {
                        total = cmp::max(total, size);
                    } else {
                        total += size;
                    }
                }

                // There needs to be no padding.
                if total != self.size {
                    None
                } else {
                    result
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CastTarget {
    Uniform(Uniform),
    Pair(Reg, Reg)
}

impl From<Reg> for CastTarget {
    fn from(unit: Reg) -> CastTarget {
        CastTarget::Uniform(Uniform::from(unit))
    }
}

impl From<Uniform> for CastTarget {
    fn from(uniform: Uniform) -> CastTarget {
        CastTarget::Uniform(uniform)
    }
}

impl CastTarget {
    pub fn size(&self, cx: &CodegenCx) -> Size {
        match *self {
            CastTarget::Uniform(u) => u.total,
            CastTarget::Pair(a, b) => {
                (a.size.abi_align(a.align(cx)) + b.size)
                    .abi_align(self.align(cx))
            }
        }
    }

    pub fn align(&self, cx: &CodegenCx) -> Align {
        match *self {
            CastTarget::Uniform(u) => u.align(cx),
            CastTarget::Pair(a, b) => {
                cx.data_layout().aggregate_align
                    .max(a.align(cx))
                    .max(b.align(cx))
            }
        }
    }

    pub fn llvm_type(&self, cx: &CodegenCx) -> Type {
        match *self {
            CastTarget::Uniform(u) => u.llvm_type(cx),
            CastTarget::Pair(a, b) => {
                Type::struct_(cx, &[
                    a.llvm_type(cx),
                    b.llvm_type(cx)
                ], false)
            }
        }
    }
}

/// Information about how to pass an argument to,
/// or return a value from, a function, under some ABI.
#[derive(Debug)]
pub struct ArgType<'tcx> {
    pub layout: TyLayout<'tcx>,

    /// Dummy argument, which is emitted before the real argument.
    pub pad: Option<Reg>,

    pub mode: PassMode,
}

impl<'a, 'tcx> ArgType<'tcx> {
    fn new(layout: TyLayout<'tcx>) -> ArgType<'tcx> {
        ArgType {
            layout,
            pad: None,
            mode: PassMode::Direct(ArgAttributes::new()),
        }
    }

    pub fn make_indirect(&mut self) {
        assert_eq!(self.mode, PassMode::Direct(ArgAttributes::new()));

        // Start with fresh attributes for the pointer.
        let mut attrs = ArgAttributes::new();

        // For non-immediate arguments the callee gets its own copy of
        // the value on the stack, so there are no aliases. It's also
        // program-invisible so can't possibly capture
        attrs.set(ArgAttribute::NoAlias)
             .set(ArgAttribute::NoCapture)
             .set(ArgAttribute::NonNull);
        attrs.pointee_size = self.layout.size;
        // FIXME(eddyb) We should be doing this, but at least on
        // i686-pc-windows-msvc, it results in wrong stack offsets.
        // attrs.pointee_align = Some(self.layout.align);

        self.mode = PassMode::Indirect(attrs);
    }

    pub fn make_indirect_byval(&mut self) {
        self.make_indirect();
        match self.mode {
            PassMode::Indirect(ref mut attrs) => {
                attrs.set(ArgAttribute::ByVal);
            }
            _ => bug!()
        }
    }

    pub fn extend_integer_width_to(&mut self, bits: u64) {
        // Only integers have signedness
        if let layout::Abi::Scalar(ref scalar) = self.layout.abi {
            if let layout::Int(i, signed) = scalar.value {
                if i.size().bits() < bits {
                    if let PassMode::Direct(ref mut attrs) = self.mode {
                        attrs.set(if signed {
                            ArgAttribute::SExt
                        } else {
                            ArgAttribute::ZExt
                        });
                    }
                }
            }
        }
    }

    pub fn cast_to<T: Into<CastTarget>>(&mut self, target: T) {
        assert_eq!(self.mode, PassMode::Direct(ArgAttributes::new()));
        self.mode = PassMode::Cast(target.into());
    }

    pub fn pad_with(&mut self, reg: Reg) {
        self.pad = Some(reg);
    }

    pub fn is_indirect(&self) -> bool {
        match self.mode {
            PassMode::Indirect(_) => true,
            _ => false
        }
    }

    pub fn is_ignore(&self) -> bool {
        self.mode == PassMode::Ignore
    }

    /// Get the LLVM type for an place of the original Rust type of
    /// this argument/return, i.e. the result of `type_of::type_of`.
    pub fn memory_ty(&self, cx: &CodegenCx<'a, 'tcx>) -> Type {
        self.layout.llvm_type(cx)
    }

    /// Store a direct/indirect value described by this ArgType into a
    /// place for the original Rust type of this argument/return.
    /// Can be used for both storing formal arguments into Rust variables
    /// or results of call/invoke instructions into their destinations.
    pub fn store(&self, bx: &Builder<'a, 'tcx>, val: ValueRef, dst: PlaceRef<'tcx>) {
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
                                  self.layout.align.min(scratch_align));

                bx.lifetime_end(llscratch, scratch_size);
            }
        } else {
            OperandValue::Immediate(val).store(bx, dst);
        }
    }

    pub fn store_fn_arg(&self, bx: &Builder<'a, 'tcx>, idx: &mut usize, dst: PlaceRef<'tcx>) {
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

/// Metadata describing how the arguments to a native function
/// should be passed in order to respect the native ABI.
///
/// I will do my best to describe this structure, but these
/// comments are reverse-engineered and may be inaccurate. -NDM
#[derive(Debug)]
pub struct FnType<'tcx> {
    /// The LLVM types of each argument.
    pub args: Vec<ArgType<'tcx>>,

    /// LLVM return type.
    pub ret: ArgType<'tcx>,

    pub variadic: bool,

    pub cconv: llvm::CallConv
}

impl<'a, 'tcx> FnType<'tcx> {
    pub fn of_instance(cx: &CodegenCx<'a, 'tcx>, instance: &ty::Instance<'tcx>)
                       -> Self {
        let fn_ty = instance.ty(cx.tcx);
        let sig = ty_fn_sig(cx, fn_ty);
        let sig = cx.tcx.erase_late_bound_regions_and_normalize(&sig);
        FnType::new(cx, sig, &[])
    }

    pub fn new(cx: &CodegenCx<'a, 'tcx>,
               sig: ty::FnSig<'tcx>,
               extra_args: &[Ty<'tcx>]) -> FnType<'tcx> {
        let mut fn_ty = FnType::unadjusted(cx, sig, extra_args);
        fn_ty.adjust_for_abi(cx, sig.abi);
        fn_ty
    }

    pub fn new_vtable(cx: &CodegenCx<'a, 'tcx>,
                      sig: ty::FnSig<'tcx>,
                      extra_args: &[Ty<'tcx>]) -> FnType<'tcx> {
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

            let pointee = self_arg.layout.ty.builtin_deref(true, ty::NoPreference)
                .unwrap_or_else(|| {
                    bug!("FnType::new_vtable: non-pointer self {:?}", self_arg)
                }).ty;
            let fat_ptr_ty = cx.tcx.mk_mut_ptr(pointee);
            self_arg.layout = cx.layout_of(fat_ptr_ty).field(cx, 0);
        }
        fn_ty.adjust_for_abi(cx, sig.abi);
        fn_ty
    }

    pub fn unadjusted(cx: &CodegenCx<'a, 'tcx>,
                      sig: ty::FnSig<'tcx>,
                      extra_args: &[Ty<'tcx>]) -> FnType<'tcx> {
        debug!("FnType::unadjusted({:?}, {:?})", sig, extra_args);

        use self::Abi::*;
        let cconv = match cx.sess().target.target.adjust_abi(sig.abi) {
            RustIntrinsic | PlatformIntrinsic |
            Rust | RustCall => llvm::CCallConv,

            // It's the ABI's job to select this, not us.
            System => bug!("system abi should be selected elsewhere"),

            Stdcall => llvm::X86StdcallCallConv,
            Fastcall => llvm::X86FastcallCallConv,
            Vectorcall => llvm::X86_VectorCall,
            Thiscall => llvm::X86_ThisCall,
            C => llvm::CCallConv,
            Unadjusted => llvm::CCallConv,
            Win64 => llvm::X86_64_Win64,
            SysV64 => llvm::X86_64_SysV,
            Aapcs => llvm::ArmAapcsCallConv,
            PtxKernel => llvm::PtxKernel,
            Msp430Interrupt => llvm::Msp430Intr,
            X86Interrupt => llvm::X86_Intr,

            // These API constants ought to be more specific...
            Cdecl => llvm::CCallConv,
        };

        let mut inputs = sig.inputs();
        let extra_args = if sig.abi == RustCall {
            assert!(!sig.variadic && extra_args.is_empty());

            match sig.inputs().last().unwrap().sty {
                ty::TyTuple(ref tupled_arguments, _) => {
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
                                      layout: TyLayout<'tcx>,
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

            if scalar.valid_range.start < scalar.valid_range.end {
                if scalar.valid_range.start > 0 {
                    attrs.set(ArgAttribute::NonNull);
                }
            }

            if let Some(pointee) = layout.pointee_info_at(cx, offset) {
                if let Some(kind) = pointee.safe {
                    attrs.pointee_size = pointee.size;
                    attrs.pointee_align = Some(pointee.align);

                    // HACK(eddyb) LLVM inserts `llvm.assume` calls when inlining functions
                    // with align attributes, and those calls later block optimizations.
                    if !is_return {
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
                                           Size::from_bytes(0),
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
                                           Size::from_bytes(0),
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
            cconv,
        }
    }

    fn adjust_for_abi(&mut self,
                      cx: &CodegenCx<'a, 'tcx>,
                      abi: Abi) {
        if abi == Abi::Unadjusted { return }

        if abi == Abi::Rust || abi == Abi::RustCall ||
           abi == Abi::RustIntrinsic || abi == Abi::PlatformIntrinsic {
            let fixup = |arg: &mut ArgType<'tcx>| {
                if arg.is_ignore() { return; }

                match arg.layout.abi {
                    layout::Abi::Aggregate { .. } => {}
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

        match &cx.sess().target.target.arch[..] {
            "x86" => {
                let flavor = if abi == Abi::Fastcall {
                    cabi_x86::Flavor::Fastcall
                } else {
                    cabi_x86::Flavor::General
                };
                cabi_x86::compute_abi_info(cx, self, flavor);
            },
            "x86_64" => if abi == Abi::SysV64 {
                cabi_x86_64::compute_abi_info(cx, self);
            } else if abi == Abi::Win64 || cx.sess().target.target.options.is_like_windows {
                cabi_x86_win64::compute_abi_info(self);
            } else {
                cabi_x86_64::compute_abi_info(cx, self);
            },
            "aarch64" => cabi_aarch64::compute_abi_info(cx, self),
            "arm" => cabi_arm::compute_abi_info(cx, self),
            "mips" => cabi_mips::compute_abi_info(cx, self),
            "mips64" => cabi_mips64::compute_abi_info(cx, self),
            "powerpc" => cabi_powerpc::compute_abi_info(cx, self),
            "powerpc64" => cabi_powerpc64::compute_abi_info(cx, self),
            "s390x" => cabi_s390x::compute_abi_info(cx, self),
            "asmjs" => cabi_asmjs::compute_abi_info(cx, self),
            "wasm32" => cabi_asmjs::compute_abi_info(cx, self),
            "msp430" => cabi_msp430::compute_abi_info(self),
            "sparc" => cabi_sparc::compute_abi_info(cx, self),
            "sparc64" => cabi_sparc64::compute_abi_info(cx, self),
            "nvptx" => cabi_nvptx::compute_abi_info(self),
            "nvptx64" => cabi_nvptx64::compute_abi_info(self),
            "hexagon" => cabi_hexagon::compute_abi_info(self),
            a => cx.sess().fatal(&format!("unrecognized arch \"{}\" in target specification", a))
        }

        if let PassMode::Indirect(ref mut attrs) = self.ret.mode {
            attrs.set(ArgAttribute::StructRet);
        }
    }

    pub fn llvm_type(&self, cx: &CodegenCx<'a, 'tcx>) -> Type {
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

    pub fn apply_attrs_llfn(&self, llfn: ValueRef) {
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

    pub fn apply_attrs_callsite(&self, callsite: ValueRef) {
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

        if self.cconv != llvm::CCallConv {
            llvm::SetInstructionCallConv(callsite, self.cconv);
        }
    }
}
