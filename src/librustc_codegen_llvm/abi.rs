use crate::llvm::{self, AttributePlace};
use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::type_::Type;
use crate::type_of::{LayoutLlvmExt, PointerKind};
use crate::value::Value;
use rustc_codegen_ssa::MemFlags;
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::mir::operand::OperandValue;
use rustc_target::abi::call::ArgType;

use rustc_codegen_ssa::traits::*;

use rustc_target::abi::{HasDataLayout, LayoutOf, Size, TyLayout, Abi as LayoutAbi};
use rustc::ty::{self, Ty, Instance};
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
    fn apply_llfn(&self, idx: AttributePlace, llfn: &Value);
    fn apply_callsite(&self, idx: AttributePlace, callsite: &Value);
}

impl ArgAttributesExt for ArgAttributes {
    fn apply_llfn(&self, idx: AttributePlace, llfn: &Value) {
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
                                               align.bytes() as u32);
            }
            regular.for_each_kind(|attr| attr.apply_llfn(idx, llfn));
        }
    }

    fn apply_callsite(&self, idx: AttributePlace, callsite: &Value) {
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
                                                       align.bytes() as u32);
            }
            regular.for_each_kind(|attr| attr.apply_callsite(idx, callsite));
        }
    }
}

pub trait LlvmType {
    fn llvm_type(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type;
}

impl LlvmType for Reg {
    fn llvm_type(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type {
        match self.kind {
            RegKind::Integer => cx.type_ix(self.size.bits()),
            RegKind::Float => {
                match self.size.bits() {
                    32 => cx.type_f32(),
                    64 => cx.type_f64(),
                    _ => bug!("unsupported float: {:?}", self)
                }
            }
            RegKind::Vector => {
                cx.type_vector(cx.type_i8(), self.size.bytes())
            }
        }
    }
}

impl LlvmType for CastTarget {
    fn llvm_type(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type {
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
                return cx.type_array(rest_ll_unit, rest_count);
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
            args.push(cx.type_ix(rem_bytes * 8));
        }

        cx.type_struct(&args, false)
    }
}

pub trait ArgTypeExt<'ll, 'tcx> {
    fn memory_ty(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type;
    fn store(
        &self,
        bx: &mut Builder<'_, 'll, 'tcx>,
        val: &'ll Value,
        dst: PlaceRef<'tcx, &'ll Value>,
    );
    fn store_fn_arg(
        &self,
        bx: &mut Builder<'_, 'll, 'tcx>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, &'ll Value>,
    );
}

impl ArgTypeExt<'ll, 'tcx> for ArgType<'tcx, Ty<'tcx>> {
    /// Gets the LLVM type for a place of the original Rust type of
    /// this argument/return, i.e., the result of `type_of::type_of`.
    fn memory_ty(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type {
        self.layout.llvm_type(cx)
    }

    /// Stores a direct/indirect value described by this ArgType into a
    /// place for the original Rust type of this argument/return.
    /// Can be used for both storing formal arguments into Rust variables
    /// or results of call/invoke instructions into their destinations.
    fn store(
        &self,
        bx: &mut Builder<'_, 'll, 'tcx>,
        val: &'ll Value,
        dst: PlaceRef<'tcx, &'ll Value>,
    ) {
        if self.is_ignore() {
            return;
        }
        if self.is_sized_indirect() {
            OperandValue::Ref(val, None, self.layout.align.abi).store(bx, dst)
        } else if self.is_unsized_indirect() {
            bug!("unsized ArgType must be handled through store_fn_arg");
        } else if let PassMode::Cast(cast) = self.mode {
            // FIXME(eddyb): Figure out when the simpler Store is safe, clang
            // uses it for i16 -> {i8, i8}, but not for i24 -> {i8, i8, i8}.
            let can_store_through_cast_ptr = false;
            if can_store_through_cast_ptr {
                let cast_ptr_llty = bx.type_ptr_to(cast.llvm_type(bx));
                let cast_dst = bx.pointercast(dst.llval, cast_ptr_llty);
                bx.store(val, cast_dst, self.layout.align.abi);
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
                let scratch_size = cast.size(bx);
                let scratch_align = cast.align(bx);
                let llscratch = bx.alloca(cast.llvm_type(bx), "abi_cast", scratch_align);
                bx.lifetime_start(llscratch, scratch_size);

                // ...where we first store the value...
                bx.store(val, llscratch, scratch_align);

                // ...and then memcpy it to the intended destination.
                bx.memcpy(
                    dst.llval,
                    self.layout.align.abi,
                    llscratch,
                    scratch_align,
                    bx.const_usize(self.layout.size.bytes()),
                    MemFlags::empty()
                );

                bx.lifetime_end(llscratch, scratch_size);
            }
        } else {
            OperandValue::Immediate(val).store(bx, dst);
        }
    }

    fn store_fn_arg(
        &self,
        bx: &mut Builder<'a, 'll, 'tcx>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, &'ll Value>,
    ) {
        let mut next = || {
            let val = llvm::get_param(bx.llfn(), *idx as c_uint);
            *idx += 1;
            val
        };
        match self.mode {
            PassMode::Ignore(_) => {}
            PassMode::Pair(..) => {
                OperandValue::Pair(next(), next()).store(bx, dst);
            }
            PassMode::Indirect(_, Some(_)) => {
                OperandValue::Ref(next(), Some(next()), self.layout.align.abi).store(bx, dst);
            }
            PassMode::Direct(_) | PassMode::Indirect(_, None) | PassMode::Cast(_) => {
                let next_arg = next();
                self.store(bx, next_arg, dst);
            }
        }
    }
}

impl ArgTypeMethods<'tcx> for Builder<'a, 'll, 'tcx> {
    fn store_fn_arg(
        &mut self,
        ty: &ArgType<'tcx, Ty<'tcx>>,
        idx: &mut usize, dst: PlaceRef<'tcx, Self::Value>
    ) {
        ty.store_fn_arg(self, idx, dst)
    }
    fn store_arg_ty(
        &mut self,
        ty: &ArgType<'tcx, Ty<'tcx>>,
        val: &'ll Value,
        dst: PlaceRef<'tcx, &'ll Value>
    ) {
        ty.store(self, val, dst)
    }
    fn memory_ty(&self, ty: &ArgType<'tcx, Ty<'tcx>>) -> &'ll Type {
        ty.memory_ty(self)
    }
}

pub trait FnTypeExt<'tcx> {
    fn of_instance(cx: &CodegenCx<'ll, 'tcx>, instance: &ty::Instance<'tcx>) -> Self;
    fn new(cx: &CodegenCx<'ll, 'tcx>,
           sig: ty::FnSig<'tcx>,
           extra_args: &[Ty<'tcx>]) -> Self;
    fn new_vtable(cx: &CodegenCx<'ll, 'tcx>,
                  sig: ty::FnSig<'tcx>,
                  extra_args: &[Ty<'tcx>]) -> Self;
    fn new_internal(
        cx: &CodegenCx<'ll, 'tcx>,
        sig: ty::FnSig<'tcx>,
        extra_args: &[Ty<'tcx>],
        mk_arg_type: impl Fn(Ty<'tcx>, Option<usize>) -> ArgType<'tcx, Ty<'tcx>>,
    ) -> Self;
    fn adjust_for_abi(&mut self,
                      cx: &CodegenCx<'ll, 'tcx>,
                      abi: Abi);
    fn llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type;
    fn ptr_to_llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type;
    fn llvm_cconv(&self) -> llvm::CallConv;
    fn apply_attrs_llfn(&self, llfn: &'ll Value);
    fn apply_attrs_callsite(&self, bx: &mut Builder<'a, 'll, 'tcx>, callsite: &'ll Value);
}

impl<'tcx> FnTypeExt<'tcx> for FnType<'tcx, Ty<'tcx>> {
    fn of_instance(cx: &CodegenCx<'ll, 'tcx>, instance: &ty::Instance<'tcx>) -> Self {
        let sig = instance.fn_sig(cx.tcx);
        let sig = cx.tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);
        FnType::new(cx, sig, &[])
    }

    fn new(cx: &CodegenCx<'ll, 'tcx>,
           sig: ty::FnSig<'tcx>,
           extra_args: &[Ty<'tcx>]) -> Self {
        FnType::new_internal(cx, sig, extra_args, |ty, _| {
            ArgType::new(cx.layout_of(ty))
        })
    }

    fn new_vtable(cx: &CodegenCx<'ll, 'tcx>,
                  sig: ty::FnSig<'tcx>,
                  extra_args: &[Ty<'tcx>]) -> Self {
        FnType::new_internal(cx, sig, extra_args, |ty, arg_idx| {
            let mut layout = cx.layout_of(ty);
            // Don't pass the vtable, it's not an argument of the virtual fn.
            // Instead, pass just the data pointer, but give it the type `*const/mut dyn Trait`
            // or `&/&mut dyn Trait` because this is special-cased elsewhere in codegen
            if arg_idx == Some(0) {
                let fat_pointer_ty = if layout.is_unsized() {
                    // unsized `self` is passed as a pointer to `self`
                    // FIXME (mikeyhew) change this to use &own if it is ever added to the language
                    cx.tcx.mk_mut_ptr(layout.ty)
                } else {
                    match layout.abi {
                        LayoutAbi::ScalarPair(..) => (),
                        _ => bug!("receiver type has unsupported layout: {:?}", layout)
                    }

                    // In the case of Rc<Self>, we need to explicitly pass a *mut RcBox<Self>
                    // with a Scalar (not ScalarPair) ABI. This is a hack that is understood
                    // elsewhere in the compiler as a method on a `dyn Trait`.
                    // To get the type `*mut RcBox<Self>`, we just keep unwrapping newtypes until we
                    // get a built-in pointer type
                    let mut fat_pointer_layout = layout;
                    'descend_newtypes: while !fat_pointer_layout.ty.is_unsafe_ptr()
                        && !fat_pointer_layout.ty.is_region_ptr()
                    {
                        'iter_fields: for i in 0..fat_pointer_layout.fields.count() {
                            let field_layout = fat_pointer_layout.field(cx, i);

                            if !field_layout.is_zst() {
                                fat_pointer_layout = field_layout;
                                continue 'descend_newtypes
                            }
                        }

                        bug!("receiver has no non-zero-sized fields {:?}", fat_pointer_layout);
                    }

                    fat_pointer_layout.ty
                };

                // we now have a type like `*mut RcBox<dyn Trait>`
                // change its layout to that of `*mut ()`, a thin pointer, but keep the same type
                // this is understood as a special case elsewhere in the compiler
                let unit_pointer_ty = cx.tcx.mk_mut_ptr(cx.tcx.mk_unit());
                layout = cx.layout_of(unit_pointer_ty);
                layout.ty = fat_pointer_ty;
            }
            ArgType::new(layout)
        })
    }

    fn new_internal(
        cx: &CodegenCx<'ll, 'tcx>,
        sig: ty::FnSig<'tcx>,
        extra_args: &[Ty<'tcx>],
        mk_arg_type: impl Fn(Ty<'tcx>, Option<usize>) -> ArgType<'tcx, Ty<'tcx>>,
    ) -> Self {
        debug!("FnType::new_internal({:?}, {:?})", sig, extra_args);

        use self::Abi::*;
        let conv = match cx.sess().target.target.adjust_abi(sig.abi) {
            RustIntrinsic | PlatformIntrinsic |
            Rust | RustCall => Conv::C,

            // It's the ABI's job to select this, not ours.
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
            AmdGpuKernel => Conv::AmdGpuKernel,

            // These API constants ought to be more specific...
            Cdecl => Conv::C,
        };

        let mut inputs = sig.inputs();
        let extra_args = if sig.abi == RustCall {
            assert!(!sig.c_variadic && extra_args.is_empty());

            match sig.inputs().last().unwrap().sty {
                ty::Tuple(tupled_arguments) => {
                    inputs = &sig.inputs()[0..sig.inputs().len() - 1];
                    tupled_arguments.iter().map(|k| k.expect_ty()).collect()
                }
                _ => {
                    bug!("argument to function with \"rust-call\" ABI \
                          is not a tuple");
                }
            }
        } else {
            assert!(sig.c_variadic || extra_args.is_empty());
            extra_args.to_vec()
        };

        let target = &cx.sess().target.target;
        let win_x64_gnu = target.target_os == "windows"
                       && target.arch == "x86_64"
                       && target.target_env == "gnu";
        let linux_s390x = target.target_os == "linux"
                       && target.arch == "s390x"
                       && target.target_env == "gnu";
        let linux_sparc64 = target.target_os == "linux"
                       && target.arch == "sparc64"
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

        // Store the index of the last argument. This is useful for working with
        // C-compatible variadic arguments.
        let last_arg_idx = if sig.inputs().is_empty() {
            None
        } else {
            Some(sig.inputs().len() - 1)
        };

        let arg_of = |ty: Ty<'tcx>, arg_idx: Option<usize>| {
            let is_return = arg_idx.is_none();
            let mut arg = mk_arg_type(ty, arg_idx);
            if arg.layout.is_zst() {
                // For some forsaken reason, x86_64-pc-windows-gnu
                // doesn't ignore zero-sized struct arguments.
                // The same is true for s390x-unknown-linux-gnu
                // and sparc64-unknown-linux-gnu.
                if is_return || rust_abi || (!win_x64_gnu && !linux_s390x && !linux_sparc64) {
                    arg.mode = PassMode::Ignore(IgnoreMode::Zst);
                }
            }

            // If this is a C-variadic function, this is not the return value,
            // and there is one or more fixed arguments; ensure that the `VaList`
            // is ignored as an argument.
            if sig.c_variadic {
                match (last_arg_idx, arg_idx) {
                    (Some(last_idx), Some(cur_idx)) if last_idx == cur_idx => {
                        let va_list_did = match cx.tcx.lang_items().va_list() {
                            Some(did) => did,
                            None => bug!("`va_list` lang item required for C-variadic functions"),
                        };
                        match ty.sty {
                            ty::Adt(def, _) if def.did == va_list_did => {
                                // This is the "spoofed" `VaList`. Set the arguments mode
                                // so that it will be ignored.
                                arg.mode = PassMode::Ignore(IgnoreMode::CVarArgs);
                            },
                            _ => (),
                        }
                    }
                    _ => {}
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
                                           a.value.size(cx).align_to(b.value.align(cx).abi),
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

        let mut fn_ty = FnType {
            ret: arg_of(sig.output(), None),
            args: inputs.iter().cloned().chain(extra_args).enumerate().map(|(i, ty)| {
                arg_of(ty, Some(i))
            }).collect(),
            c_variadic: sig.c_variadic,
            conv,
        };
        fn_ty.adjust_for_abi(cx, sig.abi);
        fn_ty
    }

    fn adjust_for_abi(&mut self,
                      cx: &CodegenCx<'ll, 'tcx>,
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
                    layout::Abi::Vector { .. }
                        if abi != Abi::PlatformIntrinsic &&
                            cx.sess().target.target.options.simd_types_indirect =>
                    {
                        arg.make_indirect();
                        return
                    }

                    _ => return
                }

                let size = arg.layout.size;
                if arg.layout.is_unsized() || size > layout::Pointer.size(cx) {
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
            if let PassMode::Indirect(ref mut attrs, _) = self.ret.mode {
                attrs.set(ArgAttribute::StructRet);
            }
            return;
        }

        if let Err(msg) = self.adjust_for_cabi(cx, abi) {
            cx.sess().fatal(&msg);
        }
    }

    fn llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type {
        let args_capacity: usize = self.args.iter().map(|arg|
            if arg.pad.is_some() { 1 } else { 0 } +
            if let PassMode::Pair(_, _) = arg.mode { 2 } else { 1 }
        ).sum();
        let mut llargument_tys = Vec::with_capacity(
            if let PassMode::Indirect(..) = self.ret.mode { 1 } else { 0 } + args_capacity
        );

        let llreturn_ty = match self.ret.mode {
            PassMode::Ignore(IgnoreMode::Zst) => cx.type_void(),
            PassMode::Ignore(IgnoreMode::CVarArgs) =>
                bug!("`va_list` should never be a return type"),
            PassMode::Direct(_) | PassMode::Pair(..) => {
                self.ret.layout.immediate_llvm_type(cx)
            }
            PassMode::Cast(cast) => cast.llvm_type(cx),
            PassMode::Indirect(..) => {
                llargument_tys.push(cx.type_ptr_to(self.ret.memory_ty(cx)));
                cx.type_void()
            }
        };

        for arg in &self.args {
            // add padding
            if let Some(ty) = arg.pad {
                llargument_tys.push(ty.llvm_type(cx));
            }

            let llarg_ty = match arg.mode {
                PassMode::Ignore(_) => continue,
                PassMode::Direct(_) => arg.layout.immediate_llvm_type(cx),
                PassMode::Pair(..) => {
                    llargument_tys.push(arg.layout.scalar_pair_element_llvm_type(cx, 0, true));
                    llargument_tys.push(arg.layout.scalar_pair_element_llvm_type(cx, 1, true));
                    continue;
                }
                PassMode::Indirect(_, Some(_)) => {
                    let ptr_ty = cx.tcx.mk_mut_ptr(arg.layout.ty);
                    let ptr_layout = cx.layout_of(ptr_ty);
                    llargument_tys.push(ptr_layout.scalar_pair_element_llvm_type(cx, 0, true));
                    llargument_tys.push(ptr_layout.scalar_pair_element_llvm_type(cx, 1, true));
                    continue;
                }
                PassMode::Cast(cast) => cast.llvm_type(cx),
                PassMode::Indirect(_, None) => cx.type_ptr_to(arg.memory_ty(cx)),
            };
            llargument_tys.push(llarg_ty);
        }

        if self.c_variadic {
            cx.type_variadic_func(&llargument_tys, llreturn_ty)
        } else {
            cx.type_func(&llargument_tys, llreturn_ty)
        }
    }

    fn ptr_to_llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type {
        unsafe {
            llvm::LLVMPointerType(self.llvm_type(cx),
                                  cx.data_layout().instruction_address_space as c_uint)
        }
    }

    fn llvm_cconv(&self) -> llvm::CallConv {
        match self.conv {
            Conv::C => llvm::CCallConv,
            Conv::AmdGpuKernel => llvm::AmdGpuKernel,
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

    fn apply_attrs_llfn(&self, llfn: &'ll Value) {
        let mut i = 0;
        let mut apply = |attrs: &ArgAttributes| {
            attrs.apply_llfn(llvm::AttributePlace::Argument(i), llfn);
            i += 1;
        };
        match self.ret.mode {
            PassMode::Direct(ref attrs) => {
                attrs.apply_llfn(llvm::AttributePlace::ReturnValue, llfn);
            }
            PassMode::Indirect(ref attrs, _) => apply(attrs),
            _ => {}
        }
        for arg in &self.args {
            if arg.pad.is_some() {
                apply(&ArgAttributes::new());
            }
            match arg.mode {
                PassMode::Ignore(_) => {}
                PassMode::Direct(ref attrs) |
                PassMode::Indirect(ref attrs, None) => apply(attrs),
                PassMode::Indirect(ref attrs, Some(ref extra_attrs)) => {
                    apply(attrs);
                    apply(extra_attrs);
                }
                PassMode::Pair(ref a, ref b) => {
                    apply(a);
                    apply(b);
                }
                PassMode::Cast(_) => apply(&ArgAttributes::new()),
            }
        }
    }

    fn apply_attrs_callsite(&self, bx: &mut Builder<'a, 'll, 'tcx>, callsite: &'ll Value) {
        let mut i = 0;
        let mut apply = |attrs: &ArgAttributes| {
            attrs.apply_callsite(llvm::AttributePlace::Argument(i), callsite);
            i += 1;
        };
        match self.ret.mode {
            PassMode::Direct(ref attrs) => {
                attrs.apply_callsite(llvm::AttributePlace::ReturnValue, callsite);
            }
            PassMode::Indirect(ref attrs, _) => apply(attrs),
            _ => {}
        }
        if let layout::Abi::Scalar(ref scalar) = self.ret.layout.abi {
            // If the value is a boolean, the range is 0..2 and that ultimately
            // become 0..0 when the type becomes i1, which would be rejected
            // by the LLVM verifier.
            if let layout::Int(..) = scalar.value {
                if !scalar.is_bool() {
                    let range = scalar.valid_range_exclusive(bx);
                    if range.start != range.end {
                        bx.range_metadata(callsite, range);
                    }
                }
            }
        }
        for arg in &self.args {
            if arg.pad.is_some() {
                apply(&ArgAttributes::new());
            }
            match arg.mode {
                PassMode::Ignore(_) => {}
                PassMode::Direct(ref attrs) |
                PassMode::Indirect(ref attrs, None) => apply(attrs),
                PassMode::Indirect(ref attrs, Some(ref extra_attrs)) => {
                    apply(attrs);
                    apply(extra_attrs);
                }
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

impl AbiMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn new_fn_type(&self, sig: ty::FnSig<'tcx>, extra_args: &[Ty<'tcx>]) -> FnType<'tcx, Ty<'tcx>> {
        FnType::new(&self, sig, extra_args)
    }
    fn new_vtable(
        &self,
        sig: ty::FnSig<'tcx>,
        extra_args: &[Ty<'tcx>]
    ) -> FnType<'tcx, Ty<'tcx>> {
        FnType::new_vtable(&self, sig, extra_args)
    }
    fn fn_type_of_instance(&self, instance: &Instance<'tcx>) -> FnType<'tcx, Ty<'tcx>> {
        FnType::of_instance(&self, instance)
    }
}

impl AbiBuilderMethods<'tcx> for Builder<'a, 'll, 'tcx> {
    fn apply_attrs_callsite(
        &mut self,
        ty: &FnType<'tcx, Ty<'tcx>>,
        callsite: Self::Value
    ) {
        ty.apply_attrs_callsite(self, callsite)
    }

    fn get_param(&self, index: usize) -> Self::Value {
        llvm::get_param(self.llfn(), index as c_uint)
    }
}
