// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{self, ValueRef, Integer, Pointer, Float, Double, Struct, Array, Vector, AttributePlace};
use base;
use builder::Builder;
use common::{type_is_fat_ptr, C_uint};
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
use machine::{llalign_of_min, llsize_of, llsize_of_alloc};
use type_::Type;
use type_of;

use rustc::hir;
use rustc::ty::{self, Ty};

use libc::c_uint;
use std::cmp;

pub use syntax::abi::Abi;
pub use rustc::ty::layout::{FAT_PTR_ADDR, FAT_PTR_EXTRA};
use rustc::ty::layout::Layout;

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

/// Information about how a specific C type
/// should be passed to or returned from a function
///
/// This is borrowed from clang's ABIInfo.h
#[derive(Clone, Copy, Debug)]
pub struct ArgType {
    kind: ArgKind,
    /// Original LLVM type
    pub original_ty: Type,
    /// Sizing LLVM type (pointers are opaque).
    /// Unlike original_ty, this is guaranteed to be complete.
    ///
    /// For example, while we're computing the function pointer type in
    /// `struct Foo(fn(Foo));`, `original_ty` is still LLVM's `%Foo = {}`.
    /// The field type will likely end up being `void(%Foo)*`, but we cannot
    /// use `%Foo` to compute properties (e.g. size and alignment) of `Foo`,
    /// until `%Foo` is completed by having all of its field types inserted,
    /// so `ty` holds the "sizing type" of `Foo`, which replaces all pointers
    /// with opaque ones, resulting in `{i8*}` for `Foo`.
    /// ABI-specific logic can then look at the size, alignment and fields of
    /// `{i8*}` in order to determine how the argument will be passed.
    /// Only later will `original_ty` aka `%Foo` be used in the LLVM function
    /// pointer type, without ever having introspected it.
    pub ty: Type,
    /// Signedness for integer types, None for other types
    pub signedness: Option<bool>,
    /// Coerced LLVM Type
    pub cast: Option<Type>,
    /// Dummy argument, which is emitted before the real argument
    pub pad: Option<Type>,
    /// LLVM attributes of argument
    pub attrs: ArgAttributes
}

impl ArgType {
    fn new(original_ty: Type, ty: Type) -> ArgType {
        ArgType {
            kind: ArgKind::Direct,
            original_ty: original_ty,
            ty: ty,
            signedness: None,
            cast: None,
            pad: None,
            attrs: ArgAttributes::default()
        }
    }

    pub fn make_indirect(&mut self, ccx: &CrateContext) {
        assert_eq!(self.kind, ArgKind::Direct);

        // Wipe old attributes, likely not valid through indirection.
        self.attrs = ArgAttributes::default();

        let llarg_sz = llsize_of_alloc(ccx, self.ty);

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
        if let Some(signed) = self.signedness {
            if self.ty.int_width() < bits {
                self.attrs.set(if signed {
                    ArgAttribute::SExt
                } else {
                    ArgAttribute::ZExt
                });
            }
        }
    }

    pub fn is_indirect(&self) -> bool {
        self.kind == ArgKind::Indirect
    }

    pub fn is_ignore(&self) -> bool {
        self.kind == ArgKind::Ignore
    }

    /// Get the LLVM type for an lvalue of the original Rust type of
    /// this argument/return, i.e. the result of `type_of::type_of`.
    pub fn memory_ty(&self, ccx: &CrateContext) -> Type {
        if self.original_ty == Type::i1(ccx) {
            Type::i8(ccx)
        } else {
            self.original_ty
        }
    }

    /// Store a direct/indirect value described by this ArgType into a
    /// lvalue for the original Rust type of this argument/return.
    /// Can be used for both storing formal arguments into Rust variables
    /// or results of call/invoke instructions into their destinations.
    pub fn store(&self, bcx: &Builder, mut val: ValueRef, dst: ValueRef) {
        if self.is_ignore() {
            return;
        }
        let ccx = bcx.ccx;
        if self.is_indirect() {
            let llsz = llsize_of(ccx, self.ty);
            let llalign = llalign_of_min(ccx, self.ty);
            base::call_memcpy(bcx, dst, val, llsz, llalign as u32);
        } else if let Some(ty) = self.cast {
            // FIXME(eddyb): Figure out when the simpler Store is safe, clang
            // uses it for i16 -> {i8, i8}, but not for i24 -> {i8, i8, i8}.
            let can_store_through_cast_ptr = false;
            if can_store_through_cast_ptr {
                let cast_dst = bcx.pointercast(dst, ty.ptr_to());
                let llalign = llalign_of_min(ccx, self.ty);
                bcx.store(val, cast_dst, Some(llalign));
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
                let llscratch = bcx.alloca(ty, "abi_cast");
                base::Lifetime::Start.call(bcx, llscratch);

                // ...where we first store the value...
                bcx.store(val, llscratch, None);

                // ...and then memcpy it to the intended destination.
                base::call_memcpy(bcx,
                                  bcx.pointercast(dst, Type::i8p(ccx)),
                                  bcx.pointercast(llscratch, Type::i8p(ccx)),
                                  C_uint(ccx, llsize_of_alloc(ccx, self.ty)),
                                  cmp::min(llalign_of_min(ccx, self.ty),
                                           llalign_of_min(ccx, ty)) as u32);

                base::Lifetime::End.call(bcx, llscratch);
            }
        } else {
            if self.original_ty == Type::i1(ccx) {
                val = bcx.zext(val, Type::i8(ccx));
            }
            bcx.store(val, dst, None);
        }
    }

    pub fn store_fn_arg(&self, bcx: &Builder, idx: &mut usize, dst: ValueRef) {
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
pub struct FnType {
    /// The LLVM types of each argument.
    pub args: Vec<ArgType>,

    /// LLVM return type.
    pub ret: ArgType,

    pub variadic: bool,

    pub cconv: llvm::CallConv
}

impl FnType {
    pub fn new<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                         abi: Abi,
                         sig: &ty::FnSig<'tcx>,
                         extra_args: &[Ty<'tcx>]) -> FnType {
        let mut fn_ty = FnType::unadjusted(ccx, abi, sig, extra_args);
        fn_ty.adjust_for_abi(ccx, abi, sig);
        fn_ty
    }

    pub fn unadjusted<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                abi: Abi,
                                sig: &ty::FnSig<'tcx>,
                                extra_args: &[Ty<'tcx>]) -> FnType {
        use self::Abi::*;
        let cconv = match ccx.sess().target.target.adjust_abi(abi) {
            RustIntrinsic | PlatformIntrinsic |
            Rust | RustCall => llvm::CCallConv,

            // It's the ABI's job to select this, not us.
            System => bug!("system abi should be selected elsewhere"),

            Stdcall => llvm::X86StdcallCallConv,
            Fastcall => llvm::X86FastcallCallConv,
            Vectorcall => llvm::X86_VectorCall,
            C => llvm::CCallConv,
            Unadjusted => llvm::CCallConv,
            Win64 => llvm::X86_64_Win64,
            SysV64 => llvm::X86_64_SysV,
            Aapcs => llvm::ArmAapcsCallConv,
            PtxKernel => llvm::PtxKernel,
            Msp430Interrupt => llvm::Msp430Intr,

            // These API constants ought to be more specific...
            Cdecl => llvm::CCallConv,
        };

        let mut inputs = sig.inputs();
        let extra_args = if abi == RustCall {
            assert!(!sig.variadic && extra_args.is_empty());

            match sig.inputs().last().unwrap().sty {
                ty::TyTuple(ref tupled_arguments) => {
                    inputs = &sig.inputs()[0..sig.inputs().len() - 1];
                    &tupled_arguments[..]
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
        let rust_abi = match abi {
            RustIntrinsic | PlatformIntrinsic | Rust | RustCall => true,
            _ => false
        };

        let arg_of = |ty: Ty<'tcx>, is_return: bool| {
            if ty.is_bool() {
                let llty = Type::i1(ccx);
                let mut arg = ArgType::new(llty, llty);
                arg.attrs.set(ArgAttribute::ZExt);
                arg
            } else {
                let mut arg = ArgType::new(type_of::type_of(ccx, ty),
                                           type_of::sizing_type_of(ccx, ty));
                if ty.is_integral() {
                    arg.signedness = Some(ty.is_signed());
                }
                // Rust enum types that map onto C enums also need to follow
                // the target ABI zero-/sign-extension rules.
                if let Layout::CEnum { signed, .. } = *ccx.layout_of(ty) {
                    arg.signedness = Some(signed);
                }
                if llsize_of_alloc(ccx, arg.ty) == 0 {
                    // For some forsaken reason, x86_64-pc-windows-gnu
                    // doesn't ignore zero-sized struct arguments.
                    // The same is true for s390x-unknown-linux-gnu.
                    if is_return || rust_abi ||
                       (!win_x64_gnu && !linux_s390x) {
                        arg.ignore();
                    }
                }
                arg
            }
        };

        let ret_ty = sig.output();
        let mut ret = arg_of(ret_ty, true);

        if !type_is_fat_ptr(ccx, ret_ty) {
            // The `noalias` attribute on the return value is useful to a
            // function ptr caller.
            if let ty::TyBox(_) = ret_ty.sty {
                // `Box` pointer return values never alias because ownership
                // is transferred
                ret.attrs.set(ArgAttribute::NoAlias);
            }

            // We can also mark the return value as `dereferenceable` in certain cases
            match ret_ty.sty {
                // These are not really pointers but pairs, (pointer, len)
                ty::TyRef(_, ty::TypeAndMut { ty, .. }) |
                ty::TyBox(ty) => {
                    let llty = type_of::sizing_type_of(ccx, ty);
                    let llsz = llsize_of_alloc(ccx, llty);
                    ret.attrs.set_dereferenceable(llsz);
                }
                _ => {}
            }
        }

        let mut args = Vec::with_capacity(inputs.len() + extra_args.len());

        // Handle safe Rust thin and fat pointers.
        let rust_ptr_attrs = |ty: Ty<'tcx>, arg: &mut ArgType| match ty.sty {
            // `Box` pointer parameters never alias because ownership is transferred
            ty::TyBox(inner) => {
                arg.attrs.set(ArgAttribute::NoAlias);
                Some(inner)
            }

            ty::TyRef(b, mt) => {
                use rustc::ty::{BrAnon, ReLateBound};

                // `&mut` pointer parameters never alias other parameters, or mutable global data
                //
                // `&T` where `T` contains no `UnsafeCell<U>` is immutable, and can be marked as
                // both `readonly` and `noalias`, as LLVM's definition of `noalias` is based solely
                // on memory dependencies rather than pointer equality
                let interior_unsafe = mt.ty.type_contents(ccx.tcx()).interior_unsafe();

                if mt.mutbl != hir::MutMutable && !interior_unsafe {
                    arg.attrs.set(ArgAttribute::NoAlias);
                }

                if mt.mutbl == hir::MutImmutable && !interior_unsafe {
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

            if type_is_fat_ptr(ccx, ty) {
                let original_tys = arg.original_ty.field_types();
                let sizing_tys = arg.ty.field_types();
                assert_eq!((original_tys.len(), sizing_tys.len()), (2, 2));

                let mut data = ArgType::new(original_tys[0], sizing_tys[0]);
                let mut info = ArgType::new(original_tys[1], sizing_tys[1]);

                if let Some(inner) = rust_ptr_attrs(ty, &mut data) {
                    data.attrs.set(ArgAttribute::NonNull);
                    if ccx.tcx().struct_tail(inner).is_trait() {
                        info.attrs.set(ArgAttribute::NonNull);
                    }
                }
                args.push(data);
                args.push(info);
            } else {
                if let Some(inner) = rust_ptr_attrs(ty, &mut arg) {
                    let llty = type_of::sizing_type_of(ccx, inner);
                    let llsz = llsize_of_alloc(ccx, llty);
                    arg.attrs.set_dereferenceable(llsz);
                }
                args.push(arg);
            }
        }

        FnType {
            args: args,
            ret: ret,
            variadic: sig.variadic,
            cconv: cconv
        }
    }

    pub fn adjust_for_abi<'a, 'tcx>(&mut self,
                                    ccx: &CrateContext<'a, 'tcx>,
                                    abi: Abi,
                                    sig: &ty::FnSig<'tcx>) {
        if abi == Abi::Unadjusted { return }

        if abi == Abi::Rust || abi == Abi::RustCall ||
           abi == Abi::RustIntrinsic || abi == Abi::PlatformIntrinsic {
            let fixup = |arg: &mut ArgType| {
                let mut llty = arg.ty;

                // Replace newtypes with their inner-most type.
                while llty.kind() == llvm::TypeKind::Struct {
                    let inner = llty.field_types();
                    if inner.len() != 1 {
                        break;
                    }
                    llty = inner[0];
                }

                if !llty.is_aggregate() {
                    // Scalars and vectors, always immediate.
                    if llty != arg.ty {
                        // Needs a cast as we've unpacked a newtype.
                        arg.cast = Some(llty);
                    }
                    return;
                }

                let size = llsize_of_alloc(ccx, llty);
                if size > llsize_of_alloc(ccx, ccx.int_type()) {
                    arg.make_indirect(ccx);
                } else if size > 0 {
                    // We want to pass small aggregates as immediates, but using
                    // a LLVM aggregate type for this leads to bad optimizations,
                    // so we pick an appropriately sized integer type instead.
                    arg.cast = Some(Type::ix(ccx, size * 8));
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
            "arm" => {
                let flavor = if ccx.sess().target.target.target_os == "ios" {
                    cabi_arm::Flavor::Ios
                } else {
                    cabi_arm::Flavor::General
                };
                cabi_arm::compute_abi_info(ccx, self, flavor);
            },
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
            a => ccx.sess().fatal(&format!("unrecognized arch \"{}\" in target specification", a))
        }

        if self.ret.is_indirect() {
            self.ret.attrs.set(ArgAttribute::StructRet);
        }
    }

    pub fn llvm_type(&self, ccx: &CrateContext) -> Type {
        let mut llargument_tys = Vec::new();

        let llreturn_ty = if self.ret.is_ignore() {
            Type::void(ccx)
        } else if self.ret.is_indirect() {
            llargument_tys.push(self.ret.original_ty.ptr_to());
            Type::void(ccx)
        } else {
            self.ret.cast.unwrap_or(self.ret.original_ty)
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
                arg.original_ty.ptr_to()
            } else {
                arg.cast.unwrap_or(arg.original_ty)
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

pub fn align_up_to(off: usize, a: usize) -> usize {
    return (off + a - 1) / a * a;
}

fn align(off: usize, ty: Type, pointer: usize) -> usize {
    let a = ty_align(ty, pointer);
    return align_up_to(off, a);
}

pub fn ty_align(ty: Type, pointer: usize) -> usize {
    match ty.kind() {
        Integer => ((ty.int_width() as usize) + 7) / 8,
        Pointer => pointer,
        Float => 4,
        Double => 8,
        Struct => {
            if ty.is_packed() {
                1
            } else {
                let str_tys = ty.field_types();
                str_tys.iter().fold(1, |a, t| cmp::max(a, ty_align(*t, pointer)))
            }
        }
        Array => {
            let elt = ty.element_type();
            ty_align(elt, pointer)
        }
        Vector => {
            let len = ty.vector_length();
            let elt = ty.element_type();
            ty_align(elt, pointer) * len
        }
        _ => bug!("ty_align: unhandled type")
    }
}

pub fn ty_size(ty: Type, pointer: usize) -> usize {
    match ty.kind() {
        Integer => ((ty.int_width() as usize) + 7) / 8,
        Pointer => pointer,
        Float => 4,
        Double => 8,
        Struct => {
            if ty.is_packed() {
                let str_tys = ty.field_types();
                str_tys.iter().fold(0, |s, t| s + ty_size(*t, pointer))
            } else {
                let str_tys = ty.field_types();
                let size = str_tys.iter().fold(0, |s, t| {
                    align(s, *t, pointer) + ty_size(*t, pointer)
                });
                align(size, ty, pointer)
            }
        }
        Array => {
            let len = ty.array_length();
            let elt = ty.element_type();
            let eltsz = ty_size(elt, pointer);
            len * eltsz
        }
        Vector => {
            let len = ty.vector_length();
            let elt = ty.element_type();
            let eltsz = ty_size(elt, pointer);
            len * eltsz
        },
        _ => bug!("ty_size: unhandled type")
    }
}
