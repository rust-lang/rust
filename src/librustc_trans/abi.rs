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
use common::{instance_ty, ty_fn_sig, type_is_fat_ptr, C_uint};
use context::CrateContext;
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
use machine::llalign_of_min;
use type_::Type;
use type_of;

use rustc::hir;
use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, Layout, LayoutTyper, TyLayout, Size};

use libc::c_uint;
use std::cmp;
use std::iter;

pub use syntax::abi::Abi;
pub use rustc::ty::layout::{FAT_PTR_ADDR, FAT_PTR_EXTRA};

#[derive(Clone, Copy, PartialEq, Debug)]
enum ArgKind {
    /// Pass the argument directly using the normal converted
    /// LLVM type or by coercing to another specified type
    Direct,
    /// Pass the argument indirectly via a hidden pointer
    Indirect,
    /// Ignore the argument (useful for empty struct)
    Ignore,
}

// Hack to disable non_upper_case_globals only for the bitflags! and not for the rest
// of this module
pub use self::attr_impl::ArgAttribute;

#[allow(non_upper_case_globals)]
#[allow(unused)]
mod attr_impl {
    // The subset of llvm::Attribute needed for arguments, packed into a bitfield.
    bitflags! {
        #[derive(Default, Debug)]
        flags ArgAttribute : u16 {
            const ByVal     = 1 << 0,
            const NoAlias   = 1 << 1,
            const NoCapture = 1 << 2,
            const NonNull   = 1 << 3,
            const ReadOnly  = 1 << 4,
            const SExt      = 1 << 5,
            const StructRet = 1 << 6,
            const ZExt      = 1 << 7,
            const InReg     = 1 << 8,
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
#[derive(Copy, Clone, Debug, Default)]
pub struct ArgAttributes {
    regular: ArgAttribute,
    dereferenceable_bytes: u64,
}

impl ArgAttributes {
    pub fn set(&mut self, attr: ArgAttribute) -> &mut Self {
        self.regular = self.regular | attr;
        self
    }

    pub fn set_dereferenceable(&mut self, bytes: u64) -> &mut Self {
        self.dereferenceable_bytes = bytes;
        self
    }

    pub fn apply_llfn(&self, idx: AttributePlace, llfn: ValueRef) {
        unsafe {
            self.regular.for_each_kind(|attr| attr.apply_llfn(idx, llfn));
            if self.dereferenceable_bytes != 0 {
                llvm::LLVMRustAddDereferenceableAttr(llfn,
                                                     idx.as_uint(),
                                                     self.dereferenceable_bytes);
            }
        }
    }

    pub fn apply_callsite(&self, idx: AttributePlace, callsite: ValueRef) {
        unsafe {
            self.regular.for_each_kind(|attr| attr.apply_callsite(idx, callsite));
            if self.dereferenceable_bytes != 0 {
                llvm::LLVMRustAddDereferenceableCallSiteAttr(callsite,
                                                             idx.as_uint(),
                                                             self.dereferenceable_bytes);
            }
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
    fn llvm_type(&self, ccx: &CrateContext) -> Type {
        match self.kind {
            RegKind::Integer => Type::ix(ccx, self.size.bits()),
            RegKind::Float => {
                match self.size.bits() {
                    32 => Type::f32(ccx),
                    64 => Type::f64(ccx),
                    _ => bug!("unsupported float: {:?}", self)
                }
            }
            RegKind::Vector => {
                Type::vector(&Type::i8(ccx), self.size.bytes())
            }
        }
    }
}

/// An argument passed entirely registers with the
/// same kind (e.g. HFA / HVA on PPC64 and AArch64).
#[derive(Copy, Clone)]
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
    fn llvm_type(&self, ccx: &CrateContext) -> Type {
        let llunit = self.unit.llvm_type(ccx);

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
            .chain(iter::once(Type::ix(ccx, rem_bytes * 8)))
            .collect();

        Type::struct_(ccx, &args, false)
    }
}

pub trait LayoutExt<'tcx> {
    fn is_aggregate(&self) -> bool;
    fn homogeneous_aggregate<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> Option<Reg>;
}

impl<'tcx> LayoutExt<'tcx> for TyLayout<'tcx> {
    fn is_aggregate(&self) -> bool {
        match *self.layout {
            Layout::Scalar { .. } |
            Layout::RawNullablePointer { .. } |
            Layout::CEnum { .. } |
            Layout::Vector { .. } => false,

            Layout::Array { .. } |
            Layout::FatPointer { .. } |
            Layout::Univariant { .. } |
            Layout::UntaggedUnion { .. } |
            Layout::General { .. } |
            Layout::StructWrappedNullablePointer { .. } => true
        }
    }

    fn homogeneous_aggregate<'a>(&self, ccx: &CrateContext<'a, 'tcx>) -> Option<Reg> {
        match *self.layout {
            // The primitives for this algorithm.
            Layout::Scalar { value, .. } |
            Layout::RawNullablePointer { value, .. } => {
                let kind = match value {
                    layout::Int(_) |
                    layout::Pointer => RegKind::Integer,
                    layout::F32 |
                    layout::F64 => RegKind::Float
                };
                Some(Reg {
                    kind,
                    size: self.size(ccx)
                })
            }

            Layout::CEnum { .. } => {
                Some(Reg {
                    kind: RegKind::Integer,
                    size: self.size(ccx)
                })
            }

            Layout::Vector { .. } => {
                Some(Reg {
                    kind: RegKind::Vector,
                    size: self.size(ccx)
                })
            }

            Layout::Array { count, .. } => {
                if count > 0 {
                    self.field(ccx, 0).homogeneous_aggregate(ccx)
                } else {
                    None
                }
            }

            Layout::Univariant { ref variant, .. } => {
                let mut unaligned_offset = Size::from_bytes(0);
                let mut result = None;

                for i in 0..self.field_count() {
                    if unaligned_offset != variant.offsets[i] {
                        return None;
                    }

                    let field = self.field(ccx, i);
                    match (result, field.homogeneous_aggregate(ccx)) {
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
                    let size = field.size(ccx);
                    match unaligned_offset.checked_add(size, ccx) {
                        Some(offset) => unaligned_offset = offset,
                        None => return None
                    }
                }

                // There needs to be no padding.
                if unaligned_offset != self.size(ccx) {
                    None
                } else {
                    result
                }
            }

            Layout::UntaggedUnion { .. } => {
                let mut max = Size::from_bytes(0);
                let mut result = None;

                for i in 0..self.field_count() {
                    let field = self.field(ccx, i);
                    match (result, field.homogeneous_aggregate(ccx)) {
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
                    let size = field.size(ccx);
                    if size > max {
                        max = size;
                    }
                }

                // There needs to be no padding.
                if max != self.size(ccx) {
                    None
                } else {
                    result
                }
            }

            // Rust-specific types, which we can ignore for C ABIs.
            Layout::FatPointer { .. } |
            Layout::General { .. } |
            Layout::StructWrappedNullablePointer { .. } => None
        }
    }
}

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
    fn llvm_type(&self, ccx: &CrateContext) -> Type {
        match *self {
            CastTarget::Uniform(u) => u.llvm_type(ccx),
            CastTarget::Pair(a, b) => {
                Type::struct_(ccx, &[
                    a.llvm_type(ccx),
                    b.llvm_type(ccx)
                ], false)
            }
        }
    }
}

/// Information about how a specific C type
/// should be passed to or returned from a function
///
/// This is borrowed from clang's ABIInfo.h
#[derive(Clone, Copy, Debug)]
pub struct ArgType<'tcx> {
    kind: ArgKind,
    pub layout: TyLayout<'tcx>,
    /// Coerced LLVM Type
    pub cast: Option<Type>,
    /// Dummy argument, which is emitted before the real argument
    pub pad: Option<Type>,
    /// LLVM attributes of argument
    pub attrs: ArgAttributes
}

impl<'a, 'tcx> ArgType<'tcx> {
    fn new(layout: TyLayout<'tcx>) -> ArgType<'tcx> {
        ArgType {
            kind: ArgKind::Direct,
            layout,
            cast: None,
            pad: None,
            attrs: ArgAttributes::default()
        }
    }

    pub fn make_indirect(&mut self, ccx: &CrateContext<'a, 'tcx>) {
        assert_eq!(self.kind, ArgKind::Direct);

        // Wipe old attributes, likely not valid through indirection.
        self.attrs = ArgAttributes::default();

        let llarg_sz = self.layout.size(ccx).bytes();

        // For non-immediate arguments the callee gets its own copy of
        // the value on the stack, so there are no aliases. It's also
        // program-invisible so can't possibly capture
        self.attrs.set(ArgAttribute::NoAlias)
                  .set(ArgAttribute::NoCapture)
                  .set_dereferenceable(llarg_sz);

        self.kind = ArgKind::Indirect;
    }

    pub fn ignore(&mut self) {
        assert_eq!(self.kind, ArgKind::Direct);
        self.kind = ArgKind::Ignore;
    }

    pub fn extend_integer_width_to(&mut self, bits: u64) {
        // Only integers have signedness
        let (i, signed) = match *self.layout {
            Layout::Scalar { value, .. } => {
                match value {
                    layout::Int(i) => {
                        if self.layout.ty.is_integral() {
                            (i, self.layout.ty.is_signed())
                        } else {
                            return;
                        }
                    }
                    _ => return
                }
            }

            // Rust enum types that map onto C enums also need to follow
            // the target ABI zero-/sign-extension rules.
            Layout::CEnum { discr, signed, .. } => (discr, signed),

            _ => return
        };

        if i.size().bits() < bits {
            self.attrs.set(if signed {
                ArgAttribute::SExt
            } else {
                ArgAttribute::ZExt
            });
        }
    }

    pub fn cast_to<T: Into<CastTarget>>(&mut self, ccx: &CrateContext, target: T) {
        self.cast = Some(target.into().llvm_type(ccx));
    }

    pub fn pad_with(&mut self, ccx: &CrateContext, reg: Reg) {
        self.pad = Some(reg.llvm_type(ccx));
    }

    pub fn is_indirect(&self) -> bool {
        self.kind == ArgKind::Indirect
    }

    pub fn is_ignore(&self) -> bool {
        self.kind == ArgKind::Ignore
    }

    /// Get the LLVM type for an lvalue of the original Rust type of
    /// this argument/return, i.e. the result of `type_of::type_of`.
    pub fn memory_ty(&self, ccx: &CrateContext<'a, 'tcx>) -> Type {
        type_of::type_of(ccx, self.layout.ty)
    }

    /// Store a direct/indirect value described by this ArgType into a
    /// lvalue for the original Rust type of this argument/return.
    /// Can be used for both storing formal arguments into Rust variables
    /// or results of call/invoke instructions into their destinations.
    pub fn store(&self, bcx: &Builder<'a, 'tcx>, mut val: ValueRef, dst: ValueRef) {
        if self.is_ignore() {
            return;
        }
        let ccx = bcx.ccx;
        if self.is_indirect() {
            let llsz = C_uint(ccx, self.layout.size(ccx).bytes());
            let llalign = self.layout.align(ccx).abi();
            base::call_memcpy(bcx, dst, val, llsz, llalign as u32);
        } else if let Some(ty) = self.cast {
            // FIXME(eddyb): Figure out when the simpler Store is safe, clang
            // uses it for i16 -> {i8, i8}, but not for i24 -> {i8, i8, i8}.
            let can_store_through_cast_ptr = false;
            if can_store_through_cast_ptr {
                let cast_dst = bcx.pointercast(dst, ty.ptr_to());
                let llalign = self.layout.align(ccx).abi();
                bcx.store(val, cast_dst, Some(llalign as u32));
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
                let llscratch = bcx.alloca(ty, "abi_cast", None);
                base::Lifetime::Start.call(bcx, llscratch);

                // ...where we first store the value...
                bcx.store(val, llscratch, None);

                // ...and then memcpy it to the intended destination.
                base::call_memcpy(bcx,
                                  bcx.pointercast(dst, Type::i8p(ccx)),
                                  bcx.pointercast(llscratch, Type::i8p(ccx)),
                                  C_uint(ccx, self.layout.size(ccx).bytes()),
                                  cmp::min(self.layout.align(ccx).abi() as u32,
                                           llalign_of_min(ccx, ty)));

                base::Lifetime::End.call(bcx, llscratch);
            }
        } else {
            if self.layout.ty == ccx.tcx().types.bool {
                val = bcx.zext(val, Type::i8(ccx));
            }
            bcx.store(val, dst, None);
        }
    }

    pub fn store_fn_arg(&self, bcx: &Builder<'a, 'tcx>, idx: &mut usize, dst: ValueRef) {
        if self.pad.is_some() {
            *idx += 1;
        }
        if self.is_ignore() {
            return;
        }
        let val = llvm::get_param(bcx.llfn(), *idx as c_uint);
        *idx += 1;
        self.store(bcx, val, dst);
    }
}

/// Metadata describing how the arguments to a native function
/// should be passed in order to respect the native ABI.
///
/// I will do my best to describe this structure, but these
/// comments are reverse-engineered and may be inaccurate. -NDM
#[derive(Clone, Debug)]
pub struct FnType<'tcx> {
    /// The LLVM types of each argument.
    pub args: Vec<ArgType<'tcx>>,

    /// LLVM return type.
    pub ret: ArgType<'tcx>,

    pub variadic: bool,

    pub cconv: llvm::CallConv
}

impl<'a, 'tcx> FnType<'tcx> {
    pub fn of_instance(ccx: &CrateContext<'a, 'tcx>, instance: &ty::Instance<'tcx>)
                       -> Self {
        let fn_ty = instance_ty(ccx.shared(), &instance);
        let sig = ty_fn_sig(ccx, fn_ty);
        let sig = ccx.tcx().erase_late_bound_regions_and_normalize(&sig);
        Self::new(ccx, sig, &[])
    }

    pub fn new(ccx: &CrateContext<'a, 'tcx>,
               sig: ty::FnSig<'tcx>,
               extra_args: &[Ty<'tcx>]) -> FnType<'tcx> {
        let mut fn_ty = FnType::unadjusted(ccx, sig, extra_args);
        fn_ty.adjust_for_abi(ccx, sig);
        fn_ty
    }

    pub fn new_vtable(ccx: &CrateContext<'a, 'tcx>,
                      sig: ty::FnSig<'tcx>,
                      extra_args: &[Ty<'tcx>]) -> FnType<'tcx> {
        let mut fn_ty = FnType::unadjusted(ccx, sig, extra_args);
        // Don't pass the vtable, it's not an argument of the virtual fn.
        fn_ty.args[1].ignore();
        fn_ty.adjust_for_abi(ccx, sig);
        fn_ty
    }

    pub fn unadjusted(ccx: &CrateContext<'a, 'tcx>,
                      sig: ty::FnSig<'tcx>,
                      extra_args: &[Ty<'tcx>]) -> FnType<'tcx> {
        debug!("FnType::unadjusted({:?}, {:?})", sig, extra_args);

        use self::Abi::*;
        let cconv = match ccx.sess().target.target.adjust_abi(sig.abi) {
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

        let target = &ccx.sess().target.target;
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

        let arg_of = |ty: Ty<'tcx>, is_return: bool| {
            let mut arg = ArgType::new(ccx.layout_of(ty));
            if ty.is_bool() {
                arg.attrs.set(ArgAttribute::ZExt);
            } else {
                if arg.layout.size(ccx).bytes() == 0 {
                    // For some forsaken reason, x86_64-pc-windows-gnu
                    // doesn't ignore zero-sized struct arguments.
                    // The same is true for s390x-unknown-linux-gnu.
                    if is_return || rust_abi ||
                       (!win_x64_gnu && !linux_s390x) {
                        arg.ignore();
                    }
                }
            }
            arg
        };

        let ret_ty = sig.output();
        let mut ret = arg_of(ret_ty, true);

        if !type_is_fat_ptr(ccx, ret_ty) {
            // The `noalias` attribute on the return value is useful to a
            // function ptr caller.
            if ret_ty.is_box() {
                // `Box` pointer return values never alias because ownership
                // is transferred
                ret.attrs.set(ArgAttribute::NoAlias);
            }

            // We can also mark the return value as `dereferenceable` in certain cases
            match ret_ty.sty {
                // These are not really pointers but pairs, (pointer, len)
                ty::TyRef(_, ty::TypeAndMut { ty, .. }) => {
                    ret.attrs.set_dereferenceable(ccx.size_of(ty));
                }
                ty::TyAdt(def, _) if def.is_box() => {
                    ret.attrs.set_dereferenceable(ccx.size_of(ret_ty.boxed_ty()));
                }
                _ => {}
            }
        }

        let mut args = Vec::with_capacity(inputs.len() + extra_args.len());

        // Handle safe Rust thin and fat pointers.
        let rust_ptr_attrs = |ty: Ty<'tcx>, arg: &mut ArgType| match ty.sty {
            // `Box` pointer parameters never alias because ownership is transferred
            ty::TyAdt(def, _) if def.is_box() => {
                arg.attrs.set(ArgAttribute::NoAlias);
                Some(ty.boxed_ty())
            }

            ty::TyRef(b, mt) => {
                use rustc::ty::{BrAnon, ReLateBound};

                // `&mut` pointer parameters never alias other parameters, or mutable global data
                //
                // `&T` where `T` contains no `UnsafeCell<U>` is immutable, and can be marked as
                // both `readonly` and `noalias`, as LLVM's definition of `noalias` is based solely
                // on memory dependencies rather than pointer equality
                let is_freeze = ccx.shared().type_is_freeze(mt.ty);

                if mt.mutbl != hir::MutMutable && is_freeze {
                    arg.attrs.set(ArgAttribute::NoAlias);
                }

                if mt.mutbl == hir::MutImmutable && is_freeze {
                    arg.attrs.set(ArgAttribute::ReadOnly);
                }

                // When a reference in an argument has no named lifetime, it's
                // impossible for that reference to escape this function
                // (returned or stored beyond the call by a closure).
                if let ReLateBound(_, BrAnon(_)) = *b {
                    arg.attrs.set(ArgAttribute::NoCapture);
                }

                Some(mt.ty)
            }
            _ => None
        };

        for ty in inputs.iter().chain(extra_args.iter()) {
            let mut arg = arg_of(ty, false);

            if let ty::layout::FatPointer { .. } = *arg.layout {
                let mut data = ArgType::new(arg.layout.field(ccx, 0));
                let mut info = ArgType::new(arg.layout.field(ccx, 1));

                if let Some(inner) = rust_ptr_attrs(ty, &mut data) {
                    data.attrs.set(ArgAttribute::NonNull);
                    if ccx.tcx().struct_tail(inner).is_trait() {
                        // vtables can be safely marked non-null, readonly
                        // and noalias.
                        info.attrs.set(ArgAttribute::NonNull);
                        info.attrs.set(ArgAttribute::ReadOnly);
                        info.attrs.set(ArgAttribute::NoAlias);
                    }
                }
                args.push(data);
                args.push(info);
            } else {
                if let Some(inner) = rust_ptr_attrs(ty, &mut arg) {
                    arg.attrs.set_dereferenceable(ccx.size_of(inner));
                }
                args.push(arg);
            }
        }

        FnType {
            args,
            ret,
            variadic: sig.variadic,
            cconv,
        }
    }

    fn adjust_for_abi(&mut self,
                      ccx: &CrateContext<'a, 'tcx>,
                      sig: ty::FnSig<'tcx>) {
        let abi = sig.abi;
        if abi == Abi::Unadjusted { return }

        if abi == Abi::Rust || abi == Abi::RustCall ||
           abi == Abi::RustIntrinsic || abi == Abi::PlatformIntrinsic {
            let fixup = |arg: &mut ArgType<'tcx>| {
                if !arg.layout.is_aggregate() {
                    return;
                }

                let size = arg.layout.size(ccx);

                if let Some(unit) = arg.layout.homogeneous_aggregate(ccx) {
                    // Replace newtypes with their inner-most type.
                    if unit.size == size {
                        // Needs a cast as we've unpacked a newtype.
                        arg.cast_to(ccx, unit);
                        return;
                    }

                    // Pairs of floats.
                    if unit.kind == RegKind::Float {
                        if unit.size.checked_mul(2, ccx) == Some(size) {
                            // FIXME(eddyb) This should be using Uniform instead of a pair,
                            // but the resulting [2 x float/double] breaks emscripten.
                            // See https://github.com/kripken/emscripten-fastcomp/issues/178.
                            arg.cast_to(ccx, CastTarget::Pair(unit, unit));
                            return;
                        }
                    }
                }

                if size > layout::Pointer.size(ccx) {
                    arg.make_indirect(ccx);
                } else {
                    // We want to pass small aggregates as immediates, but using
                    // a LLVM aggregate type for this leads to bad optimizations,
                    // so we pick an appropriately sized integer type instead.
                    arg.cast_to(ccx, Reg {
                        kind: RegKind::Integer,
                        size
                    });
                }
            };
            // Fat pointers are returned by-value.
            if !self.ret.is_ignore() {
                if !type_is_fat_ptr(ccx, sig.output()) {
                    fixup(&mut self.ret);
                }
            }
            for arg in &mut self.args {
                if arg.is_ignore() { continue; }
                fixup(arg);
            }
            if self.ret.is_indirect() {
                self.ret.attrs.set(ArgAttribute::StructRet);
            }
            return;
        }

        match &ccx.sess().target.target.arch[..] {
            "x86" => {
                let flavor = if abi == Abi::Fastcall {
                    cabi_x86::Flavor::Fastcall
                } else {
                    cabi_x86::Flavor::General
                };
                cabi_x86::compute_abi_info(ccx, self, flavor);
            },
            "x86_64" => if abi == Abi::SysV64 {
                cabi_x86_64::compute_abi_info(ccx, self);
            } else if abi == Abi::Win64 || ccx.sess().target.target.options.is_like_windows {
                cabi_x86_win64::compute_abi_info(ccx, self);
            } else {
                cabi_x86_64::compute_abi_info(ccx, self);
            },
            "aarch64" => cabi_aarch64::compute_abi_info(ccx, self),
            "arm" => cabi_arm::compute_abi_info(ccx, self),
            "mips" => cabi_mips::compute_abi_info(ccx, self),
            "mips64" => cabi_mips64::compute_abi_info(ccx, self),
            "powerpc" => cabi_powerpc::compute_abi_info(ccx, self),
            "powerpc64" => cabi_powerpc64::compute_abi_info(ccx, self),
            "s390x" => cabi_s390x::compute_abi_info(ccx, self),
            "asmjs" => cabi_asmjs::compute_abi_info(ccx, self),
            "wasm32" => cabi_asmjs::compute_abi_info(ccx, self),
            "msp430" => cabi_msp430::compute_abi_info(ccx, self),
            "sparc" => cabi_sparc::compute_abi_info(ccx, self),
            "sparc64" => cabi_sparc64::compute_abi_info(ccx, self),
            "nvptx" => cabi_nvptx::compute_abi_info(ccx, self),
            "nvptx64" => cabi_nvptx64::compute_abi_info(ccx, self),
            "hexagon" => cabi_hexagon::compute_abi_info(ccx, self),
            a => ccx.sess().fatal(&format!("unrecognized arch \"{}\" in target specification", a))
        }

        if self.ret.is_indirect() {
            self.ret.attrs.set(ArgAttribute::StructRet);
        }
    }

    pub fn llvm_type(&self, ccx: &CrateContext<'a, 'tcx>) -> Type {
        let mut llargument_tys = Vec::new();

        let llreturn_ty = if self.ret.is_ignore() {
            Type::void(ccx)
        } else if self.ret.is_indirect() {
            llargument_tys.push(self.ret.memory_ty(ccx).ptr_to());
            Type::void(ccx)
        } else {
            self.ret.cast.unwrap_or_else(|| {
                type_of::immediate_type_of(ccx, self.ret.layout.ty)
            })
        };

        for arg in &self.args {
            if arg.is_ignore() {
                continue;
            }
            // add padding
            if let Some(ty) = arg.pad {
                llargument_tys.push(ty);
            }

            let llarg_ty = if arg.is_indirect() {
                arg.memory_ty(ccx).ptr_to()
            } else {
                arg.cast.unwrap_or_else(|| {
                    type_of::immediate_type_of(ccx, arg.layout.ty)
                })
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
        let mut i = if self.ret.is_indirect() { 1 } else { 0 };
        if !self.ret.is_ignore() {
            self.ret.attrs.apply_llfn(llvm::AttributePlace::Argument(i), llfn);
        }
        i += 1;
        for arg in &self.args {
            if !arg.is_ignore() {
                if arg.pad.is_some() { i += 1; }
                arg.attrs.apply_llfn(llvm::AttributePlace::Argument(i), llfn);
                i += 1;
            }
        }
    }

    pub fn apply_attrs_callsite(&self, callsite: ValueRef) {
        let mut i = if self.ret.is_indirect() { 1 } else { 0 };
        if !self.ret.is_ignore() {
            self.ret.attrs.apply_callsite(llvm::AttributePlace::Argument(i), callsite);
        }
        i += 1;
        for arg in &self.args {
            if !arg.is_ignore() {
                if arg.pad.is_some() { i += 1; }
                arg.attrs.apply_callsite(llvm::AttributePlace::Argument(i), callsite);
                i += 1;
            }
        }

        if self.cconv != llvm::CCallConv {
            llvm::SetInstructionCallConv(callsite, self.cconv);
        }
    }
}

pub fn align_up_to(off: u64, a: u64) -> u64 {
    (off + a - 1) / a * a
}
