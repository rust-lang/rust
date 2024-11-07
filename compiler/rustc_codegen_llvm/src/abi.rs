use std::borrow::Borrow;
use std::cmp;

use libc::c_uint;
use rustc_abi::{BackendRepr, HasDataLayout, Primitive, Reg, RegKind, Size};
use rustc_codegen_ssa::MemFlags;
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::{PlaceRef, PlaceValue};
use rustc_codegen_ssa::traits::*;
use rustc_middle::ty::Ty;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::{bug, ty};
use rustc_session::config;
use rustc_target::callconv::{
    ArgAbi, ArgAttribute, ArgAttributes, ArgExtension, CastTarget, Conv, FnAbi, PassMode,
};
use rustc_target::spec::SanitizerSet;
use smallvec::SmallVec;

use crate::attributes::{self, llfn_attrs_from_instance};
use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::llvm::{self, Attribute, AttributePlace};
use crate::type_::Type;
use crate::type_of::LayoutLlvmExt;
use crate::value::Value;

trait ArgAttributesExt {
    fn apply_attrs_to_llfn(&self, idx: AttributePlace, cx: &CodegenCx<'_, '_>, llfn: &Value);
    fn apply_attrs_to_callsite(
        &self,
        idx: AttributePlace,
        cx: &CodegenCx<'_, '_>,
        callsite: &Value,
    );
}

const ABI_AFFECTING_ATTRIBUTES: [(ArgAttribute, llvm::AttributeKind); 1] =
    [(ArgAttribute::InReg, llvm::AttributeKind::InReg)];

const OPTIMIZATION_ATTRIBUTES: [(ArgAttribute, llvm::AttributeKind); 5] = [
    (ArgAttribute::NoAlias, llvm::AttributeKind::NoAlias),
    (ArgAttribute::NoCapture, llvm::AttributeKind::NoCapture),
    (ArgAttribute::NonNull, llvm::AttributeKind::NonNull),
    (ArgAttribute::ReadOnly, llvm::AttributeKind::ReadOnly),
    (ArgAttribute::NoUndef, llvm::AttributeKind::NoUndef),
];

fn get_attrs<'ll>(this: &ArgAttributes, cx: &CodegenCx<'ll, '_>) -> SmallVec<[&'ll Attribute; 8]> {
    let mut regular = this.regular;

    let mut attrs = SmallVec::new();

    // ABI-affecting attributes must always be applied
    for (attr, llattr) in ABI_AFFECTING_ATTRIBUTES {
        if regular.contains(attr) {
            attrs.push(llattr.create_attr(cx.llcx));
        }
    }
    if let Some(align) = this.pointee_align {
        attrs.push(llvm::CreateAlignmentAttr(cx.llcx, align.bytes()));
    }
    match this.arg_ext {
        ArgExtension::None => {}
        ArgExtension::Zext => attrs.push(llvm::AttributeKind::ZExt.create_attr(cx.llcx)),
        ArgExtension::Sext => attrs.push(llvm::AttributeKind::SExt.create_attr(cx.llcx)),
    }

    // Only apply remaining attributes when optimizing
    if cx.sess().opts.optimize != config::OptLevel::No {
        let deref = this.pointee_size.bytes();
        if deref != 0 {
            if regular.contains(ArgAttribute::NonNull) {
                attrs.push(llvm::CreateDereferenceableAttr(cx.llcx, deref));
            } else {
                attrs.push(llvm::CreateDereferenceableOrNullAttr(cx.llcx, deref));
            }
            regular -= ArgAttribute::NonNull;
        }
        for (attr, llattr) in OPTIMIZATION_ATTRIBUTES {
            if regular.contains(attr) {
                attrs.push(llattr.create_attr(cx.llcx));
            }
        }
    } else if cx.tcx.sess.opts.unstable_opts.sanitizer.contains(SanitizerSet::MEMORY) {
        // If we're not optimising, *but* memory sanitizer is on, emit noundef, since it affects
        // memory sanitizer's behavior.

        if regular.contains(ArgAttribute::NoUndef) {
            attrs.push(llvm::AttributeKind::NoUndef.create_attr(cx.llcx));
        }
    }

    attrs
}

impl ArgAttributesExt for ArgAttributes {
    fn apply_attrs_to_llfn(&self, idx: AttributePlace, cx: &CodegenCx<'_, '_>, llfn: &Value) {
        let attrs = get_attrs(self, cx);
        attributes::apply_to_llfn(llfn, idx, &attrs);
    }

    fn apply_attrs_to_callsite(
        &self,
        idx: AttributePlace,
        cx: &CodegenCx<'_, '_>,
        callsite: &Value,
    ) {
        let attrs = get_attrs(self, cx);
        attributes::apply_to_callsite(callsite, idx, &attrs);
    }
}

pub(crate) trait LlvmType {
    fn llvm_type<'ll>(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type;
}

impl LlvmType for Reg {
    fn llvm_type<'ll>(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type {
        match self.kind {
            RegKind::Integer => cx.type_ix(self.size.bits()),
            RegKind::Float => match self.size.bits() {
                16 => cx.type_f16(),
                32 => cx.type_f32(),
                64 => cx.type_f64(),
                128 => cx.type_f128(),
                _ => bug!("unsupported float: {:?}", self),
            },
            RegKind::Vector => cx.type_vector(cx.type_i8(), self.size.bytes()),
        }
    }
}

impl LlvmType for CastTarget {
    fn llvm_type<'ll>(&self, cx: &CodegenCx<'ll, '_>) -> &'ll Type {
        let rest_ll_unit = self.rest.unit.llvm_type(cx);
        let rest_count = if self.rest.total == Size::ZERO {
            0
        } else {
            assert_ne!(
                self.rest.unit.size,
                Size::ZERO,
                "total size {:?} cannot be divided into units of zero size",
                self.rest.total
            );
            if self.rest.total.bytes() % self.rest.unit.size.bytes() != 0 {
                assert_eq!(self.rest.unit.kind, RegKind::Integer, "only int regs can be split");
            }
            self.rest.total.bytes().div_ceil(self.rest.unit.size.bytes())
        };

        // Simplify to a single unit or an array if there's no prefix.
        // This produces the same layout, but using a simpler type.
        if self.prefix.iter().all(|x| x.is_none()) {
            // We can't do this if is_consecutive is set and the unit would get
            // split on the target. Currently, this is only relevant for i128
            // registers.
            if rest_count == 1 && (!self.rest.is_consecutive || self.rest.unit != Reg::i128()) {
                return rest_ll_unit;
            }

            return cx.type_array(rest_ll_unit, rest_count);
        }

        // Generate a struct type with the prefix and the "rest" arguments.
        let prefix_args =
            self.prefix.iter().flat_map(|option_reg| option_reg.map(|reg| reg.llvm_type(cx)));
        let rest_args = (0..rest_count).map(|_| rest_ll_unit);
        let args: Vec<_> = prefix_args.chain(rest_args).collect();
        cx.type_struct(&args, false)
    }
}

trait ArgAbiExt<'ll, 'tcx> {
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

impl<'ll, 'tcx> ArgAbiExt<'ll, 'tcx> for ArgAbi<'tcx, Ty<'tcx>> {
    /// Stores a direct/indirect value described by this ArgAbi into a
    /// place for the original Rust type of this argument/return.
    /// Can be used for both storing formal arguments into Rust variables
    /// or results of call/invoke instructions into their destinations.
    fn store(
        &self,
        bx: &mut Builder<'_, 'll, 'tcx>,
        val: &'ll Value,
        dst: PlaceRef<'tcx, &'ll Value>,
    ) {
        match &self.mode {
            PassMode::Ignore => {}
            // Sized indirect arguments
            PassMode::Indirect { attrs, meta_attrs: None, on_stack: _ } => {
                let align = attrs.pointee_align.unwrap_or(self.layout.align.abi);
                OperandValue::Ref(PlaceValue::new_sized(val, align)).store(bx, dst);
            }
            // Unsized indirect qrguments
            PassMode::Indirect { attrs: _, meta_attrs: Some(_), on_stack: _ } => {
                bug!("unsized `ArgAbi` must be handled through `store_fn_arg`");
            }
            PassMode::Cast { cast, pad_i32: _ } => {
                // The ABI mandates that the value is passed as a different struct representation.
                // Spill and reload it from the stack to convert from the ABI representation to
                // the Rust representation.
                let scratch_size = cast.size(bx);
                let scratch_align = cast.align(bx);
                // Note that the ABI type may be either larger or smaller than the Rust type,
                // due to the presence or absence of trailing padding. For example:
                // - On some ABIs, the Rust layout { f64, f32, <f32 padding> } may omit padding
                //   when passed by value, making it smaller.
                // - On some ABIs, the Rust layout { u16, u16, u16 } may be padded up to 8 bytes
                //   when passed by value, making it larger.
                let copy_bytes =
                    cmp::min(cast.unaligned_size(bx).bytes(), self.layout.size.bytes());
                // Allocate some scratch space...
                let llscratch = bx.alloca(scratch_size, scratch_align);
                bx.lifetime_start(llscratch, scratch_size);
                // ...store the value...
                bx.store(val, llscratch, scratch_align);
                // ... and then memcpy it to the intended destination.
                bx.memcpy(
                    dst.val.llval,
                    self.layout.align.abi,
                    llscratch,
                    scratch_align,
                    bx.const_usize(copy_bytes),
                    MemFlags::empty(),
                );
                bx.lifetime_end(llscratch, scratch_size);
            }
            _ => {
                OperandRef::from_immediate_or_packed_pair(bx, val, self.layout).val.store(bx, dst);
            }
        }
    }

    fn store_fn_arg(
        &self,
        bx: &mut Builder<'_, 'll, 'tcx>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, &'ll Value>,
    ) {
        let mut next = || {
            let val = llvm::get_param(bx.llfn(), *idx as c_uint);
            *idx += 1;
            val
        };
        match self.mode {
            PassMode::Ignore => {}
            PassMode::Pair(..) => {
                OperandValue::Pair(next(), next()).store(bx, dst);
            }
            PassMode::Indirect { attrs: _, meta_attrs: Some(_), on_stack: _ } => {
                let place_val = PlaceValue {
                    llval: next(),
                    llextra: Some(next()),
                    align: self.layout.align.abi,
                };
                OperandValue::Ref(place_val).store(bx, dst);
            }
            PassMode::Direct(_)
            | PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: _ }
            | PassMode::Cast { .. } => {
                let next_arg = next();
                self.store(bx, next_arg, dst);
            }
        }
    }
}

impl<'ll, 'tcx> ArgAbiBuilderMethods<'tcx> for Builder<'_, 'll, 'tcx> {
    fn store_fn_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, Self::Value>,
    ) {
        arg_abi.store_fn_arg(self, idx, dst)
    }
    fn store_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        val: &'ll Value,
        dst: PlaceRef<'tcx, &'ll Value>,
    ) {
        arg_abi.store(self, val, dst)
    }
}

pub(crate) trait FnAbiLlvmExt<'ll, 'tcx> {
    fn llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type;
    fn ptr_to_llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type;
    fn llvm_cconv(&self, cx: &CodegenCx<'ll, 'tcx>) -> llvm::CallConv;

    /// Apply attributes to a function declaration/definition.
    fn apply_attrs_llfn(
        &self,
        cx: &CodegenCx<'ll, 'tcx>,
        llfn: &'ll Value,
        instance: Option<ty::Instance<'tcx>>,
    );

    /// Apply attributes to a function call.
    fn apply_attrs_callsite(&self, bx: &mut Builder<'_, 'll, 'tcx>, callsite: &'ll Value);
}

impl<'ll, 'tcx> FnAbiLlvmExt<'ll, 'tcx> for FnAbi<'tcx, Ty<'tcx>> {
    fn llvm_type(&self, cx: &CodegenCx<'ll, 'tcx>) -> &'ll Type {
        // Ignore "extra" args from the call site for C variadic functions.
        // Only the "fixed" args are part of the LLVM function signature.
        let args =
            if self.c_variadic { &self.args[..self.fixed_count as usize] } else { &self.args };

        // This capacity calculation is approximate.
        let mut llargument_tys = Vec::with_capacity(
            self.args.len() + if let PassMode::Indirect { .. } = self.ret.mode { 1 } else { 0 },
        );

        let llreturn_ty = match &self.ret.mode {
            PassMode::Ignore => cx.type_void(),
            PassMode::Direct(_) | PassMode::Pair(..) => self.ret.layout.immediate_llvm_type(cx),
            PassMode::Cast { cast, pad_i32: _ } => cast.llvm_type(cx),
            PassMode::Indirect { .. } => {
                llargument_tys.push(cx.type_ptr());
                cx.type_void()
            }
        };

        for arg in args {
            // Note that the exact number of arguments pushed here is carefully synchronized with
            // code all over the place, both in the codegen_llvm and codegen_ssa crates. That's how
            // other code then knows which LLVM argument(s) correspond to the n-th Rust argument.
            let llarg_ty = match &arg.mode {
                PassMode::Ignore => continue,
                PassMode::Direct(_) => {
                    // ABI-compatible Rust types have the same `layout.abi` (up to validity ranges),
                    // and for Scalar ABIs the LLVM type is fully determined by `layout.abi`,
                    // guaranteeing that we generate ABI-compatible LLVM IR.
                    arg.layout.immediate_llvm_type(cx)
                }
                PassMode::Pair(..) => {
                    // ABI-compatible Rust types have the same `layout.abi` (up to validity ranges),
                    // so for ScalarPair we can easily be sure that we are generating ABI-compatible
                    // LLVM IR.
                    llargument_tys.push(arg.layout.scalar_pair_element_llvm_type(cx, 0, true));
                    llargument_tys.push(arg.layout.scalar_pair_element_llvm_type(cx, 1, true));
                    continue;
                }
                PassMode::Indirect { attrs: _, meta_attrs: Some(_), on_stack: _ } => {
                    // Construct the type of a (wide) pointer to `ty`, and pass its two fields.
                    // Any two ABI-compatible unsized types have the same metadata type and
                    // moreover the same metadata value leads to the same dynamic size and
                    // alignment, so this respects ABI compatibility.
                    let ptr_ty = Ty::new_mut_ptr(cx.tcx, arg.layout.ty);
                    let ptr_layout = cx.layout_of(ptr_ty);
                    llargument_tys.push(ptr_layout.scalar_pair_element_llvm_type(cx, 0, true));
                    llargument_tys.push(ptr_layout.scalar_pair_element_llvm_type(cx, 1, true));
                    continue;
                }
                PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: _ } => cx.type_ptr(),
                PassMode::Cast { cast, pad_i32 } => {
                    // add padding
                    if *pad_i32 {
                        llargument_tys.push(Reg::i32().llvm_type(cx));
                    }
                    // Compute the LLVM type we use for this function from the cast type.
                    // We assume here that ABI-compatible Rust types have the same cast type.
                    cast.llvm_type(cx)
                }
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
        cx.type_ptr_ext(cx.data_layout().instruction_address_space)
    }

    fn llvm_cconv(&self, cx: &CodegenCx<'ll, 'tcx>) -> llvm::CallConv {
        llvm::CallConv::from_conv(self.conv, cx.tcx.sess.target.arch.borrow())
    }

    fn apply_attrs_llfn(
        &self,
        cx: &CodegenCx<'ll, 'tcx>,
        llfn: &'ll Value,
        instance: Option<ty::Instance<'tcx>>,
    ) {
        let mut func_attrs = SmallVec::<[_; 3]>::new();
        if self.ret.layout.is_uninhabited() {
            func_attrs.push(llvm::AttributeKind::NoReturn.create_attr(cx.llcx));
        }
        if !self.can_unwind {
            func_attrs.push(llvm::AttributeKind::NoUnwind.create_attr(cx.llcx));
        }
        if let Conv::RiscvInterrupt { kind } = self.conv {
            func_attrs.push(llvm::CreateAttrStringValue(cx.llcx, "interrupt", kind.as_str()));
        }
        if let Conv::CCmseNonSecureEntry = self.conv {
            func_attrs.push(llvm::CreateAttrString(cx.llcx, "cmse_nonsecure_entry"))
        }
        attributes::apply_to_llfn(llfn, llvm::AttributePlace::Function, &{ func_attrs });

        let mut i = 0;
        let mut apply = |attrs: &ArgAttributes| {
            attrs.apply_attrs_to_llfn(llvm::AttributePlace::Argument(i), cx, llfn);
            i += 1;
            i - 1
        };

        let apply_range_attr = |idx: AttributePlace, scalar: rustc_abi::Scalar| {
            if cx.sess().opts.optimize != config::OptLevel::No
                && matches!(scalar.primitive(), Primitive::Int(..))
                // If the value is a boolean, the range is 0..2 and that ultimately
                // become 0..0 when the type becomes i1, which would be rejected
                // by the LLVM verifier.
                && !scalar.is_bool()
                // LLVM also rejects full range.
                && !scalar.is_always_valid(cx)
            {
                attributes::apply_to_llfn(
                    llfn,
                    idx,
                    &[llvm::CreateRangeAttr(cx.llcx, scalar.size(cx), scalar.valid_range(cx))],
                );
            }
        };

        match &self.ret.mode {
            PassMode::Direct(attrs) => {
                attrs.apply_attrs_to_llfn(llvm::AttributePlace::ReturnValue, cx, llfn);
                if let BackendRepr::Scalar(scalar) = self.ret.layout.backend_repr {
                    apply_range_attr(llvm::AttributePlace::ReturnValue, scalar);
                }
            }
            PassMode::Indirect { attrs, meta_attrs: _, on_stack } => {
                assert!(!on_stack);
                let i = apply(attrs);
                let sret = llvm::CreateStructRetAttr(
                    cx.llcx,
                    cx.type_array(cx.type_i8(), self.ret.layout.size.bytes()),
                );
                attributes::apply_to_llfn(llfn, llvm::AttributePlace::Argument(i), &[sret]);
                if cx.sess().opts.optimize != config::OptLevel::No {
                    attributes::apply_to_llfn(
                        llfn,
                        llvm::AttributePlace::Argument(i),
                        &[
                            llvm::AttributeKind::Writable.create_attr(cx.llcx),
                            llvm::AttributeKind::DeadOnUnwind.create_attr(cx.llcx),
                        ],
                    );
                }
            }
            PassMode::Cast { cast, pad_i32: _ } => {
                cast.attrs.apply_attrs_to_llfn(llvm::AttributePlace::ReturnValue, cx, llfn);
            }
            _ => {}
        }
        for arg in self.args.iter() {
            match &arg.mode {
                PassMode::Ignore => {}
                PassMode::Indirect { attrs, meta_attrs: None, on_stack: true } => {
                    let i = apply(attrs);
                    let byval = llvm::CreateByValAttr(
                        cx.llcx,
                        cx.type_array(cx.type_i8(), arg.layout.size.bytes()),
                    );
                    attributes::apply_to_llfn(llfn, llvm::AttributePlace::Argument(i), &[byval]);
                }
                PassMode::Direct(attrs) => {
                    let i = apply(attrs);
                    if let BackendRepr::Scalar(scalar) = arg.layout.backend_repr {
                        apply_range_attr(llvm::AttributePlace::Argument(i), scalar);
                    }
                }
                PassMode::Indirect { attrs, meta_attrs: None, on_stack: false } => {
                    apply(attrs);
                }
                PassMode::Indirect { attrs, meta_attrs: Some(meta_attrs), on_stack } => {
                    assert!(!on_stack);
                    apply(attrs);
                    apply(meta_attrs);
                }
                PassMode::Pair(a, b) => {
                    let i = apply(a);
                    let ii = apply(b);
                    if let BackendRepr::ScalarPair(scalar_a, scalar_b) = arg.layout.backend_repr {
                        apply_range_attr(llvm::AttributePlace::Argument(i), scalar_a);
                        apply_range_attr(llvm::AttributePlace::Argument(ii), scalar_b);
                    }
                }
                PassMode::Cast { cast, pad_i32 } => {
                    if *pad_i32 {
                        apply(&ArgAttributes::new());
                    }
                    apply(&cast.attrs);
                }
            }
        }

        // If the declaration has an associated instance, compute extra attributes based on that.
        if let Some(instance) = instance {
            llfn_attrs_from_instance(cx, llfn, instance);
        }
    }

    fn apply_attrs_callsite(&self, bx: &mut Builder<'_, 'll, 'tcx>, callsite: &'ll Value) {
        let mut func_attrs = SmallVec::<[_; 2]>::new();
        if self.ret.layout.is_uninhabited() {
            func_attrs.push(llvm::AttributeKind::NoReturn.create_attr(bx.cx.llcx));
        }
        if !self.can_unwind {
            func_attrs.push(llvm::AttributeKind::NoUnwind.create_attr(bx.cx.llcx));
        }
        attributes::apply_to_callsite(callsite, llvm::AttributePlace::Function, &{ func_attrs });

        let mut i = 0;
        let mut apply = |cx: &CodegenCx<'_, '_>, attrs: &ArgAttributes| {
            attrs.apply_attrs_to_callsite(llvm::AttributePlace::Argument(i), cx, callsite);
            i += 1;
            i - 1
        };
        match &self.ret.mode {
            PassMode::Direct(attrs) => {
                attrs.apply_attrs_to_callsite(llvm::AttributePlace::ReturnValue, bx.cx, callsite);
            }
            PassMode::Indirect { attrs, meta_attrs: _, on_stack } => {
                assert!(!on_stack);
                let i = apply(bx.cx, attrs);
                let sret = llvm::CreateStructRetAttr(
                    bx.cx.llcx,
                    bx.cx.type_array(bx.cx.type_i8(), self.ret.layout.size.bytes()),
                );
                attributes::apply_to_callsite(callsite, llvm::AttributePlace::Argument(i), &[sret]);
            }
            PassMode::Cast { cast, pad_i32: _ } => {
                cast.attrs.apply_attrs_to_callsite(
                    llvm::AttributePlace::ReturnValue,
                    bx.cx,
                    callsite,
                );
            }
            _ => {}
        }
        for arg in self.args.iter() {
            match &arg.mode {
                PassMode::Ignore => {}
                PassMode::Indirect { attrs, meta_attrs: None, on_stack: true } => {
                    let i = apply(bx.cx, attrs);
                    let byval = llvm::CreateByValAttr(
                        bx.cx.llcx,
                        bx.cx.type_array(bx.cx.type_i8(), arg.layout.size.bytes()),
                    );
                    attributes::apply_to_callsite(
                        callsite,
                        llvm::AttributePlace::Argument(i),
                        &[byval],
                    );
                }
                PassMode::Direct(attrs)
                | PassMode::Indirect { attrs, meta_attrs: None, on_stack: false } => {
                    apply(bx.cx, attrs);
                }
                PassMode::Indirect { attrs, meta_attrs: Some(meta_attrs), on_stack: _ } => {
                    apply(bx.cx, attrs);
                    apply(bx.cx, meta_attrs);
                }
                PassMode::Pair(a, b) => {
                    apply(bx.cx, a);
                    apply(bx.cx, b);
                }
                PassMode::Cast { cast, pad_i32 } => {
                    if *pad_i32 {
                        apply(bx.cx, &ArgAttributes::new());
                    }
                    apply(bx.cx, &cast.attrs);
                }
            }
        }

        let cconv = self.llvm_cconv(&bx.cx);
        if cconv != llvm::CCallConv {
            llvm::SetInstructionCallConv(callsite, cconv);
        }

        if self.conv == Conv::CCmseNonSecureCall {
            // This will probably get ignored on all targets but those supporting the TrustZone-M
            // extension (thumbv8m targets).
            let cmse_nonsecure_call = llvm::CreateAttrString(bx.cx.llcx, "cmse_nonsecure_call");
            attributes::apply_to_callsite(
                callsite,
                llvm::AttributePlace::Function,
                &[cmse_nonsecure_call],
            );
        }

        // Some intrinsics require that an elementtype attribute (with the pointee type of a
        // pointer argument) is added to the callsite.
        let element_type_index = unsafe { llvm::LLVMRustGetElementTypeArgIndex(callsite) };
        if element_type_index >= 0 {
            let arg_ty = self.args[element_type_index as usize].layout.ty;
            let pointee_ty = arg_ty.builtin_deref(true).expect("Must be pointer argument");
            let element_type_attr = unsafe {
                llvm::LLVMRustCreateElementTypeAttr(bx.llcx, bx.layout_of(pointee_ty).llvm_type(bx))
            };
            attributes::apply_to_callsite(
                callsite,
                llvm::AttributePlace::Argument(element_type_index as u32),
                &[element_type_attr],
            );
        }
    }
}

impl AbiBuilderMethods for Builder<'_, '_, '_> {
    fn get_param(&mut self, index: usize) -> Self::Value {
        llvm::get_param(self.llfn(), index as c_uint)
    }
}

impl llvm::CallConv {
    pub(crate) fn from_conv(conv: Conv, arch: &str) -> Self {
        match conv {
            Conv::C
            | Conv::Rust
            | Conv::CCmseNonSecureCall
            | Conv::CCmseNonSecureEntry
            | Conv::RiscvInterrupt { .. } => llvm::CCallConv,
            Conv::Cold => llvm::ColdCallConv,
            Conv::PreserveMost => llvm::PreserveMost,
            Conv::PreserveAll => llvm::PreserveAll,
            Conv::GpuKernel => {
                if arch == "amdgpu" {
                    llvm::AmdgpuKernel
                } else if arch == "nvptx64" {
                    llvm::PtxKernel
                } else {
                    panic!("Architecture {arch} does not support GpuKernel calling convention");
                }
            }
            Conv::AvrInterrupt => llvm::AvrInterrupt,
            Conv::AvrNonBlockingInterrupt => llvm::AvrNonBlockingInterrupt,
            Conv::ArmAapcs => llvm::ArmAapcsCallConv,
            Conv::Msp430Intr => llvm::Msp430Intr,
            Conv::X86Fastcall => llvm::X86FastcallCallConv,
            Conv::X86Intr => llvm::X86_Intr,
            Conv::X86Stdcall => llvm::X86StdcallCallConv,
            Conv::X86ThisCall => llvm::X86_ThisCall,
            Conv::X86VectorCall => llvm::X86_VectorCall,
            Conv::X86_64SysV => llvm::X86_64_SysV,
            Conv::X86_64Win64 => llvm::X86_64_Win64,
        }
    }
}
