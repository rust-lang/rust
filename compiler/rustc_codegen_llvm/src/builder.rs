use std::borrow::Cow;
use std::ops::Deref;
use std::{iter, ptr};

use libc::{c_char, c_uint};
use rustc_abi as abi;
use rustc_abi::{Align, Size, WrappingRange};
use rustc_ast::expand::autodiff_attrs::{AutoDiffAttrs, DiffActivity, DiffMode};
use rustc_codegen_ssa::MemFlags;
use rustc_codegen_ssa::common::{IntPredicate, RealPredicate, SynchronizationScope, TypeKind};
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_hir::def_id::DefId;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::ty::layout::{
    FnAbiError, FnAbiOfHelpers, FnAbiRequest, HasTypingEnv, LayoutError, LayoutOfHelpers,
    TyAndLayout,
};
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_sanitizers::{cfi, kcfi};
use rustc_session::config::OptLevel;
use rustc_span::Span;
use rustc_target::callconv::FnAbi;
use rustc_target::spec::{HasTargetSpec, SanitizerSet, Target};
use smallvec::SmallVec;
use tracing::{debug, instrument, trace};

use crate::abi::FnAbiLlvmExt;
use crate::attributes;
use crate::common::Funclet;
use crate::context::CodegenCx;
use crate::llvm::AttributePlace::Function;
use crate::llvm::{self, AtomicOrdering, AtomicRmwBinOp, BasicBlock, False, Metadata, True};
use crate::type_::Type;
use crate::type_of::LayoutLlvmExt;
use crate::value::Value;

fn get_params(fnc: &Value) -> Vec<&Value> {
    unsafe {
        let param_num = llvm::LLVMCountParams(fnc) as usize;
        let mut fnc_args: Vec<&Value> = vec![];
        fnc_args.reserve(param_num);
        llvm::LLVMGetParams(fnc, fnc_args.as_mut_ptr());
        fnc_args.set_len(param_num);
        fnc_args
    }
}

/// When differentiating `fn_to_diff`, take a `outer_fn` and generate another
/// function with expected naming and calling conventions[^1] which will be
/// discovered by the enzyme LLVM pass and its body populated with the differentiated
/// `fn_to_diff`. `outer_fn` is then modified to have a call to the generated
/// function and handle the differences between the Rust calling convention and
/// Enzyme.
/// [^1]: <https://enzyme.mit.edu/getting_started/CallingConvention/>
// FIXME(ZuseZ4): `outer_fn` should include upstream safety checks to
// cover some assumptions of enzyme/autodiff, which could lead to UB otherwise.
pub(crate) fn generate_enzyme_call<'ll>(
    llmod: &'ll llvm::Module,
    llcx: &'ll llvm::Context,
    fn_to_diff: &'ll Value,
    outer_fn: &'ll Value,
    attrs: AutoDiffAttrs,
) {
    let inputs = attrs.input_activity;
    let output = attrs.ret_activity;

    // We have to pick the name depending on whether we want forward or reverse mode autodiff.
    // FIXME(ZuseZ4): The new pass based approach should not need the {Forward/Reverse}First method anymore, since
    // it will handle higher-order derivatives correctly automatically (in theory). Currently
    // higher-order derivatives fail, so we should debug that before adjusting this code.
    let mut ad_name: String = match attrs.mode {
        DiffMode::Forward => "__enzyme_fwddiff",
        DiffMode::Reverse => "__enzyme_autodiff",
        DiffMode::ForwardFirst => "__enzyme_fwddiff",
        DiffMode::ReverseFirst => "__enzyme_autodiff",
        _ => panic!("logic bug in autodiff, unrecognized mode"),
    }
    .to_string();

    // add outer_fn name to ad_name to make it unique, in case users apply autodiff to multiple
    // functions. Unwrap will only panic, if LLVM gave us an invalid string.
    let name = llvm::get_value_name(outer_fn);
    let outer_fn_name = std::ffi::CStr::from_bytes_with_nul(name).unwrap().to_str().unwrap();
    ad_name.push_str(outer_fn_name.to_string().as_str());

    // Let us assume the user wrote the following function square:
    //
    // ```llvm
    // define double @square(double %x) {
    // entry:
    //  %0 = fmul double %x, %x
    //  ret double %0
    // }
    // ```
    //
    // The user now applies autodiff to the function square, in which case fn_to_diff will be `square`.
    // Our macro generates the following placeholder code (slightly simplified):
    //
    // ```llvm
    // define double @dsquare(double %x) {
    //  ; placeholder code
    //  return 0.0;
    // }
    // ```
    //
    // so our `outer_fn` will be `dsquare`. The unsafe code section below now removes the placeholder
    // code and inserts an autodiff call. We also add a declaration for the __enzyme_autodiff call.
    // Again, the arguments to all functions are slightly simplified.
    // ```llvm
    // declare double @__enzyme_autodiff_square(...)
    //
    // define double @dsquare(double %x) {
    // entry:
    //   %0 = tail call double (...) @__enzyme_autodiff_square(double (double)* nonnull @square, double %x)
    //   ret double %0
    // }
    // ```
    unsafe {
        // On LLVM-IR, we can luckily declare __enzyme_ functions without specifying the input
        // arguments. We do however need to declare them with their correct return type.
        // We already figured the correct return type out in our frontend, when generating the outer_fn,
        // so we can now just go ahead and use that. FIXME(ZuseZ4): This doesn't handle sret yet.
        let fn_ty = llvm::LLVMGlobalGetValueType(outer_fn);
        let ret_ty = llvm::LLVMGetReturnType(fn_ty);

        // LLVM can figure out the input types on it's own, so we take a shortcut here.
        let enzyme_ty = llvm::LLVMFunctionType(ret_ty, ptr::null(), 0, True);
        let ad_fn = llvm::LLVMRustGetOrInsertFunction(
            llmod,
            ad_name.as_ptr() as *const c_char,
            ad_name.len().try_into().unwrap(),
            enzyme_ty,
        );
        // Otherwise LLVM might inline our temporary code before the enzyme pass has a chance to
        // do it's work.
        let attr = llvm::AttributeKind::NoInline.create_attr(llcx);
        attributes::apply_to_llfn(ad_fn, Function, &[attr]);

        // first, remove all calls from fnc
        let entry = llvm::LLVMGetFirstBasicBlock(outer_fn);
        let br = llvm::LLVMRustGetTerminator(entry);
        llvm::LLVMRustEraseInstFromParent(br);

        let builder = llvm::LLVMCreateBuilderInContext(llcx);
        let last_inst = llvm::LLVMRustGetLastInstruction(entry).unwrap();
        llvm::LLVMPositionBuilderAtEnd(builder, entry);

        let num_args = llvm::LLVMCountParams(&fn_to_diff);
        let mut args = Vec::with_capacity(num_args as usize + 1);
        args.push(fn_to_diff);

        let enzyme_const = llvm::create_md_string(llcx, "enzyme_const");
        let enzyme_out = llvm::create_md_string(llcx, "enzyme_out");
        let enzyme_dup = llvm::create_md_string(llcx, "enzyme_dup");
        let enzyme_dupnoneed = llvm::create_md_string(llcx, "enzyme_dupnoneed");
        let enzyme_primal_ret = llvm::create_md_string(llcx, "enzyme_primal_return");

        match output {
            DiffActivity::Dual => {
                args.push(llvm::LLVMMetadataAsValue(llcx, enzyme_primal_ret));
            }
            DiffActivity::Active => {
                args.push(llvm::LLVMMetadataAsValue(llcx, enzyme_primal_ret));
            }
            _ => {}
        }

        trace!("matching autodiff arguments");
        // We now handle the issue that Rust level arguments not always match the llvm-ir level
        // arguments. A slice, `&[f32]`, for example, is represented as a pointer and a length on
        // llvm-ir level. The number of activities matches the number of Rust level arguments, so we
        // need to match those.
        let mut outer_pos: usize = 0;
        let mut activity_pos = 0;
        let outer_args: Vec<&llvm::Value> = get_params(outer_fn);
        while activity_pos < inputs.len() {
            let activity = inputs[activity_pos as usize];
            // Duplicated arguments received a shadow argument, into which enzyme will write the
            // gradient.
            let (activity, duplicated): (&Metadata, bool) = match activity {
                DiffActivity::None => panic!("not a valid input activity"),
                DiffActivity::Const => (enzyme_const, false),
                DiffActivity::Active => (enzyme_out, false),
                DiffActivity::ActiveOnly => (enzyme_out, false),
                DiffActivity::Dual => (enzyme_dup, true),
                DiffActivity::DualOnly => (enzyme_dupnoneed, true),
                DiffActivity::Duplicated => (enzyme_dup, true),
                DiffActivity::DuplicatedOnly => (enzyme_dupnoneed, true),
                DiffActivity::FakeActivitySize => (enzyme_const, false),
            };
            let outer_arg = outer_args[outer_pos];
            args.push(llvm::LLVMMetadataAsValue(llcx, activity));
            args.push(outer_arg);
            if duplicated {
                // We know that duplicated args by construction have a following argument,
                // so this can not be out of bounds.
                let next_outer_arg = outer_args[outer_pos + 1];
                let next_outer_ty = llvm::LLVMTypeOf(next_outer_arg);
                // FIXME(ZuseZ4): We should add support for Vec here too, but it's less urgent since
                // vectors behind references (&Vec<T>) are already supported. Users can not pass a
                // Vec by value for reverse mode, so this would only help forward mode autodiff.
                let slice = {
                    if activity_pos + 1 >= inputs.len() {
                        // If there is no arg following our ptr, it also can't be a slice,
                        // since that would lead to a ptr, int pair.
                        false
                    } else {
                        let next_activity = inputs[activity_pos + 1];
                        // We analyze the MIR types and add this dummy activity if we visit a slice.
                        next_activity == DiffActivity::FakeActivitySize
                    }
                };
                if slice {
                    // A duplicated slice will have the following two outer_fn arguments:
                    // (..., ptr1, int1, ptr2, int2, ...). We add the following llvm-ir to our __enzyme call:
                    // (..., metadata! enzyme_dup, ptr, ptr, int1, ...).
                    // FIXME(ZuseZ4): We will upstream a safety check later which asserts that
                    // int2 >= int1, which means the shadow vector is large enough to store the gradient.
                    assert!(llvm::LLVMRustGetTypeKind(next_outer_ty) == llvm::TypeKind::Integer);
                    let next_outer_arg2 = outer_args[outer_pos + 2];
                    let next_outer_ty2 = llvm::LLVMTypeOf(next_outer_arg2);
                    assert!(llvm::LLVMRustGetTypeKind(next_outer_ty2) == llvm::TypeKind::Pointer);
                    let next_outer_arg3 = outer_args[outer_pos + 3];
                    let next_outer_ty3 = llvm::LLVMTypeOf(next_outer_arg3);
                    assert!(llvm::LLVMRustGetTypeKind(next_outer_ty3) == llvm::TypeKind::Integer);
                    args.push(next_outer_arg2);
                    args.push(llvm::LLVMMetadataAsValue(llcx, enzyme_const));
                    args.push(next_outer_arg);
                    outer_pos += 4;
                    activity_pos += 2;
                } else {
                    // A duplicated pointer will have the following two outer_fn arguments:
                    // (..., ptr, ptr, ...). We add the following llvm-ir to our __enzyme call:
                    // (..., metadata! enzyme_dup, ptr, ptr, ...).
                    assert!(llvm::LLVMRustGetTypeKind(next_outer_ty) == llvm::TypeKind::Pointer);
                    args.push(next_outer_arg);
                    outer_pos += 2;
                    activity_pos += 1;
                }
            } else {
                // We do not differentiate with resprect to this argument.
                // We already added the metadata and argument above, so just increase the counters.
                outer_pos += 1;
                activity_pos += 1;
            }
        }

        let call = llvm::LLVMBuildCall2(
            builder,
            enzyme_ty,
            ad_fn,
            args.as_mut_ptr(),
            args.len().try_into().unwrap(),
            ad_name.as_ptr() as *const c_char,
        );

        // This part is a bit iffy. LLVM requires that a call to an inlineable function has some
        // metadata attachted to it, but we just created this code oota. Given that the
        // differentiated function already has partly confusing metadata, and given that this
        // affects nothing but the auttodiff IR, we take a shortcut and just steal metadata from the
        // dummy code which we inserted at a higher level.
        // FIXME(ZuseZ4): Work with Enzyme core devs to clarify what debug metadata issues we have,
        // and how to best improve it for enzyme core and rust-enzyme.
        let md_ty = llvm::LLVMGetMDKindIDInContext(
            llcx,
            "dbg".as_ptr() as *const c_char,
            "dbg".len() as c_uint,
        );
        if llvm::LLVMRustHasMetadata(last_inst, md_ty) {
            let md = llvm::LLVMRustDIGetInstMetadata(last_inst);
            let md_todiff = llvm::LLVMMetadataAsValue(llcx, md);
            llvm::LLVMSetMetadata(call, md_ty, md_todiff);
        } else {
            // We don't panic, since depending on whether we are in debug or release mode, we might
            // have no debug info to copy, which would then be ok.
            trace!("no dbg info");
        }
        // Now that we copied the metadata, get rid of dummy code.
        llvm::LLVMRustEraseInstBefore(entry, last_inst);
        llvm::LLVMRustEraseInstFromParent(last_inst);

        let void_ty = llvm::LLVMVoidTypeInContext(llcx);
        if llvm::LLVMTypeOf(call) != void_ty {
            llvm::LLVMBuildRet(builder, call);
        } else {
            llvm::LLVMBuildRetVoid(builder);
        };
        llvm::LLVMDisposeBuilder(builder);

        // Let's crash in case that we messed something up above and generated invalid IR.
        llvm::LLVMVerifyFunction(outer_fn, llvm::LLVMVerifierFailureAction::LLVMAbortProcessAction);
    }
}

// All Builders must have an llfn associated with them
#[must_use]
pub(crate) struct Builder<'a, 'll, 'tcx> {
    pub llbuilder: &'ll mut llvm::Builder<'ll>,
    pub cx: &'a CodegenCx<'ll, 'tcx>,
}

impl Drop for Builder<'_, '_, '_> {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMDisposeBuilder(&mut *(self.llbuilder as *mut _));
        }
    }
}

/// Empty string, to be used where LLVM expects an instruction name, indicating
/// that the instruction is to be left unnamed (i.e. numbered, in textual IR).
// FIXME(eddyb) pass `&CStr` directly to FFI once it's a thin pointer.
const UNNAMED: *const c_char = c"".as_ptr();

impl<'ll, 'tcx> BackendTypes for Builder<'_, 'll, 'tcx> {
    type Value = <CodegenCx<'ll, 'tcx> as BackendTypes>::Value;
    type Metadata = <CodegenCx<'ll, 'tcx> as BackendTypes>::Metadata;
    type Function = <CodegenCx<'ll, 'tcx> as BackendTypes>::Function;
    type BasicBlock = <CodegenCx<'ll, 'tcx> as BackendTypes>::BasicBlock;
    type Type = <CodegenCx<'ll, 'tcx> as BackendTypes>::Type;
    type Funclet = <CodegenCx<'ll, 'tcx> as BackendTypes>::Funclet;

    type DIScope = <CodegenCx<'ll, 'tcx> as BackendTypes>::DIScope;
    type DILocation = <CodegenCx<'ll, 'tcx> as BackendTypes>::DILocation;
    type DIVariable = <CodegenCx<'ll, 'tcx> as BackendTypes>::DIVariable;
}

impl abi::HasDataLayout for Builder<'_, '_, '_> {
    fn data_layout(&self) -> &abi::TargetDataLayout {
        self.cx.data_layout()
    }
}

impl<'tcx> ty::layout::HasTyCtxt<'tcx> for Builder<'_, '_, 'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.cx.tcx
    }
}

impl<'tcx> ty::layout::HasTypingEnv<'tcx> for Builder<'_, '_, 'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.cx.typing_env()
    }
}

impl HasTargetSpec for Builder<'_, '_, '_> {
    #[inline]
    fn target_spec(&self) -> &Target {
        self.cx.target_spec()
    }
}

impl<'tcx> LayoutOfHelpers<'tcx> for Builder<'_, '_, 'tcx> {
    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        self.cx.handle_layout_err(err, span, ty)
    }
}

impl<'tcx> FnAbiOfHelpers<'tcx> for Builder<'_, '_, 'tcx> {
    #[inline]
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> ! {
        self.cx.handle_fn_abi_err(err, span, fn_abi_request)
    }
}

impl<'ll, 'tcx> Deref for Builder<'_, 'll, 'tcx> {
    type Target = CodegenCx<'ll, 'tcx>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.cx
    }
}

macro_rules! math_builder_methods {
    ($($name:ident($($arg:ident),*) => $llvm_capi:ident),+ $(,)?) => {
        $(fn $name(&mut self, $($arg: &'ll Value),*) -> &'ll Value {
            unsafe {
                llvm::$llvm_capi(self.llbuilder, $($arg,)* UNNAMED)
            }
        })+
    }
}

macro_rules! set_math_builder_methods {
    ($($name:ident($($arg:ident),*) => ($llvm_capi:ident, $llvm_set_math:ident)),+ $(,)?) => {
        $(fn $name(&mut self, $($arg: &'ll Value),*) -> &'ll Value {
            unsafe {
                let instr = llvm::$llvm_capi(self.llbuilder, $($arg,)* UNNAMED);
                llvm::$llvm_set_math(instr);
                instr
            }
        })+
    }
}

impl<'a, 'll, 'tcx> BuilderMethods<'a, 'tcx> for Builder<'a, 'll, 'tcx> {
    type CodegenCx = CodegenCx<'ll, 'tcx>;

    fn build(cx: &'a CodegenCx<'ll, 'tcx>, llbb: &'ll BasicBlock) -> Self {
        let bx = Builder::with_cx(cx);
        unsafe {
            llvm::LLVMPositionBuilderAtEnd(bx.llbuilder, llbb);
        }
        bx
    }

    fn cx(&self) -> &CodegenCx<'ll, 'tcx> {
        self.cx
    }

    fn llbb(&self) -> &'ll BasicBlock {
        unsafe { llvm::LLVMGetInsertBlock(self.llbuilder) }
    }

    fn set_span(&mut self, _span: Span) {}

    fn append_block(cx: &'a CodegenCx<'ll, 'tcx>, llfn: &'ll Value, name: &str) -> &'ll BasicBlock {
        unsafe {
            let name = SmallCStr::new(name);
            llvm::LLVMAppendBasicBlockInContext(cx.llcx, llfn, name.as_ptr())
        }
    }

    fn append_sibling_block(&mut self, name: &str) -> &'ll BasicBlock {
        Self::append_block(self.cx, self.llfn(), name)
    }

    fn switch_to_block(&mut self, llbb: Self::BasicBlock) {
        *self = Self::build(self.cx, llbb)
    }

    fn ret_void(&mut self) {
        unsafe {
            llvm::LLVMBuildRetVoid(self.llbuilder);
        }
    }

    fn ret(&mut self, v: &'ll Value) {
        unsafe {
            llvm::LLVMBuildRet(self.llbuilder, v);
        }
    }

    fn br(&mut self, dest: &'ll BasicBlock) {
        unsafe {
            llvm::LLVMBuildBr(self.llbuilder, dest);
        }
    }

    fn cond_br(
        &mut self,
        cond: &'ll Value,
        then_llbb: &'ll BasicBlock,
        else_llbb: &'ll BasicBlock,
    ) {
        unsafe {
            llvm::LLVMBuildCondBr(self.llbuilder, cond, then_llbb, else_llbb);
        }
    }

    fn switch(
        &mut self,
        v: &'ll Value,
        else_llbb: &'ll BasicBlock,
        cases: impl ExactSizeIterator<Item = (u128, &'ll BasicBlock)>,
    ) {
        let switch =
            unsafe { llvm::LLVMBuildSwitch(self.llbuilder, v, else_llbb, cases.len() as c_uint) };
        for (on_val, dest) in cases {
            let on_val = self.const_uint_big(self.val_ty(v), on_val);
            unsafe { llvm::LLVMAddCase(switch, on_val, dest) }
        }
    }

    fn invoke(
        &mut self,
        llty: &'ll Type,
        fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        llfn: &'ll Value,
        args: &[&'ll Value],
        then: &'ll BasicBlock,
        catch: &'ll BasicBlock,
        funclet: Option<&Funclet<'ll>>,
        instance: Option<Instance<'tcx>>,
    ) -> &'ll Value {
        debug!("invoke {:?} with args ({:?})", llfn, args);

        let args = self.check_call("invoke", llty, llfn, args);
        let funclet_bundle = funclet.map(|funclet| funclet.bundle());
        let mut bundles: SmallVec<[_; 2]> = SmallVec::new();
        if let Some(funclet_bundle) = funclet_bundle {
            bundles.push(funclet_bundle);
        }

        // Emit CFI pointer type membership test
        self.cfi_type_test(fn_attrs, fn_abi, instance, llfn);

        // Emit KCFI operand bundle
        let kcfi_bundle = self.kcfi_operand_bundle(fn_attrs, fn_abi, instance, llfn);
        if let Some(kcfi_bundle) = kcfi_bundle.as_deref() {
            bundles.push(kcfi_bundle);
        }

        let invoke = unsafe {
            llvm::LLVMBuildInvokeWithOperandBundles(
                self.llbuilder,
                llty,
                llfn,
                args.as_ptr(),
                args.len() as c_uint,
                then,
                catch,
                bundles.as_ptr(),
                bundles.len() as c_uint,
                UNNAMED,
            )
        };
        if let Some(fn_abi) = fn_abi {
            fn_abi.apply_attrs_callsite(self, invoke);
        }
        invoke
    }

    fn unreachable(&mut self) {
        unsafe {
            llvm::LLVMBuildUnreachable(self.llbuilder);
        }
    }

    math_builder_methods! {
        add(a, b) => LLVMBuildAdd,
        fadd(a, b) => LLVMBuildFAdd,
        sub(a, b) => LLVMBuildSub,
        fsub(a, b) => LLVMBuildFSub,
        mul(a, b) => LLVMBuildMul,
        fmul(a, b) => LLVMBuildFMul,
        udiv(a, b) => LLVMBuildUDiv,
        exactudiv(a, b) => LLVMBuildExactUDiv,
        sdiv(a, b) => LLVMBuildSDiv,
        exactsdiv(a, b) => LLVMBuildExactSDiv,
        fdiv(a, b) => LLVMBuildFDiv,
        urem(a, b) => LLVMBuildURem,
        srem(a, b) => LLVMBuildSRem,
        frem(a, b) => LLVMBuildFRem,
        shl(a, b) => LLVMBuildShl,
        lshr(a, b) => LLVMBuildLShr,
        ashr(a, b) => LLVMBuildAShr,
        and(a, b) => LLVMBuildAnd,
        or(a, b) => LLVMBuildOr,
        xor(a, b) => LLVMBuildXor,
        neg(x) => LLVMBuildNeg,
        fneg(x) => LLVMBuildFNeg,
        not(x) => LLVMBuildNot,
        unchecked_sadd(x, y) => LLVMBuildNSWAdd,
        unchecked_uadd(x, y) => LLVMBuildNUWAdd,
        unchecked_ssub(x, y) => LLVMBuildNSWSub,
        unchecked_usub(x, y) => LLVMBuildNUWSub,
        unchecked_smul(x, y) => LLVMBuildNSWMul,
        unchecked_umul(x, y) => LLVMBuildNUWMul,
    }

    set_math_builder_methods! {
        fadd_fast(x, y) => (LLVMBuildFAdd, LLVMRustSetFastMath),
        fsub_fast(x, y) => (LLVMBuildFSub, LLVMRustSetFastMath),
        fmul_fast(x, y) => (LLVMBuildFMul, LLVMRustSetFastMath),
        fdiv_fast(x, y) => (LLVMBuildFDiv, LLVMRustSetFastMath),
        frem_fast(x, y) => (LLVMBuildFRem, LLVMRustSetFastMath),
        fadd_algebraic(x, y) => (LLVMBuildFAdd, LLVMRustSetAlgebraicMath),
        fsub_algebraic(x, y) => (LLVMBuildFSub, LLVMRustSetAlgebraicMath),
        fmul_algebraic(x, y) => (LLVMBuildFMul, LLVMRustSetAlgebraicMath),
        fdiv_algebraic(x, y) => (LLVMBuildFDiv, LLVMRustSetAlgebraicMath),
        frem_algebraic(x, y) => (LLVMBuildFRem, LLVMRustSetAlgebraicMath),
    }

    fn checked_binop(
        &mut self,
        oop: OverflowOp,
        ty: Ty<'_>,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> (Self::Value, Self::Value) {
        use rustc_middle::ty::IntTy::*;
        use rustc_middle::ty::UintTy::*;
        use rustc_middle::ty::{Int, Uint};

        let new_kind = match ty.kind() {
            Int(t @ Isize) => Int(t.normalize(self.tcx.sess.target.pointer_width)),
            Uint(t @ Usize) => Uint(t.normalize(self.tcx.sess.target.pointer_width)),
            t @ (Uint(_) | Int(_)) => *t,
            _ => panic!("tried to get overflow intrinsic for op applied to non-int type"),
        };

        let name = match oop {
            OverflowOp::Add => match new_kind {
                Int(I8) => "llvm.sadd.with.overflow.i8",
                Int(I16) => "llvm.sadd.with.overflow.i16",
                Int(I32) => "llvm.sadd.with.overflow.i32",
                Int(I64) => "llvm.sadd.with.overflow.i64",
                Int(I128) => "llvm.sadd.with.overflow.i128",

                Uint(U8) => "llvm.uadd.with.overflow.i8",
                Uint(U16) => "llvm.uadd.with.overflow.i16",
                Uint(U32) => "llvm.uadd.with.overflow.i32",
                Uint(U64) => "llvm.uadd.with.overflow.i64",
                Uint(U128) => "llvm.uadd.with.overflow.i128",

                _ => unreachable!(),
            },
            OverflowOp::Sub => match new_kind {
                Int(I8) => "llvm.ssub.with.overflow.i8",
                Int(I16) => "llvm.ssub.with.overflow.i16",
                Int(I32) => "llvm.ssub.with.overflow.i32",
                Int(I64) => "llvm.ssub.with.overflow.i64",
                Int(I128) => "llvm.ssub.with.overflow.i128",

                Uint(_) => {
                    // Emit sub and icmp instead of llvm.usub.with.overflow. LLVM considers these
                    // to be the canonical form. It will attempt to reform llvm.usub.with.overflow
                    // in the backend if profitable.
                    let sub = self.sub(lhs, rhs);
                    let cmp = self.icmp(IntPredicate::IntULT, lhs, rhs);
                    return (sub, cmp);
                }

                _ => unreachable!(),
            },
            OverflowOp::Mul => match new_kind {
                Int(I8) => "llvm.smul.with.overflow.i8",
                Int(I16) => "llvm.smul.with.overflow.i16",
                Int(I32) => "llvm.smul.with.overflow.i32",
                Int(I64) => "llvm.smul.with.overflow.i64",
                Int(I128) => "llvm.smul.with.overflow.i128",

                Uint(U8) => "llvm.umul.with.overflow.i8",
                Uint(U16) => "llvm.umul.with.overflow.i16",
                Uint(U32) => "llvm.umul.with.overflow.i32",
                Uint(U64) => "llvm.umul.with.overflow.i64",
                Uint(U128) => "llvm.umul.with.overflow.i128",

                _ => unreachable!(),
            },
        };

        let res = self.call_intrinsic(name, &[lhs, rhs]);
        (self.extract_value(res, 0), self.extract_value(res, 1))
    }

    fn from_immediate(&mut self, val: Self::Value) -> Self::Value {
        if self.cx().val_ty(val) == self.cx().type_i1() {
            self.zext(val, self.cx().type_i8())
        } else {
            val
        }
    }

    fn to_immediate_scalar(&mut self, val: Self::Value, scalar: abi::Scalar) -> Self::Value {
        if scalar.is_bool() {
            return self.trunc(val, self.cx().type_i1());
        }
        val
    }

    fn alloca(&mut self, size: Size, align: Align) -> &'ll Value {
        let mut bx = Builder::with_cx(self.cx);
        bx.position_at_start(unsafe { llvm::LLVMGetFirstBasicBlock(self.llfn()) });
        let ty = self.cx().type_array(self.cx().type_i8(), size.bytes());
        unsafe {
            let alloca = llvm::LLVMBuildAlloca(bx.llbuilder, ty, UNNAMED);
            llvm::LLVMSetAlignment(alloca, align.bytes() as c_uint);
            alloca
        }
    }

    fn dynamic_alloca(&mut self, size: &'ll Value, align: Align) -> &'ll Value {
        unsafe {
            let alloca =
                llvm::LLVMBuildArrayAlloca(self.llbuilder, self.cx().type_i8(), size, UNNAMED);
            llvm::LLVMSetAlignment(alloca, align.bytes() as c_uint);
            alloca
        }
    }

    fn load(&mut self, ty: &'ll Type, ptr: &'ll Value, align: Align) -> &'ll Value {
        unsafe {
            let load = llvm::LLVMBuildLoad2(self.llbuilder, ty, ptr, UNNAMED);
            llvm::LLVMSetAlignment(load, align.bytes() as c_uint);
            load
        }
    }

    fn volatile_load(&mut self, ty: &'ll Type, ptr: &'ll Value) -> &'ll Value {
        unsafe {
            let load = llvm::LLVMBuildLoad2(self.llbuilder, ty, ptr, UNNAMED);
            llvm::LLVMSetVolatile(load, llvm::True);
            load
        }
    }

    fn atomic_load(
        &mut self,
        ty: &'ll Type,
        ptr: &'ll Value,
        order: rustc_codegen_ssa::common::AtomicOrdering,
        size: Size,
    ) -> &'ll Value {
        unsafe {
            let load = llvm::LLVMRustBuildAtomicLoad(
                self.llbuilder,
                ty,
                ptr,
                UNNAMED,
                AtomicOrdering::from_generic(order),
            );
            // LLVM requires the alignment of atomic loads to be at least the size of the type.
            llvm::LLVMSetAlignment(load, size.bytes() as c_uint);
            load
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn load_operand(&mut self, place: PlaceRef<'tcx, &'ll Value>) -> OperandRef<'tcx, &'ll Value> {
        if place.layout.is_unsized() {
            let tail = self.tcx.struct_tail_for_codegen(place.layout.ty, self.typing_env());
            if matches!(tail.kind(), ty::Foreign(..)) {
                // Unsized locals and, at least conceptually, even unsized arguments must be copied
                // around, which requires dynamically determining their size. Therefore, we cannot
                // allow `extern` types here. Consult t-opsem before removing this check.
                panic!("unsized locals must not be `extern` types");
            }
        }
        assert_eq!(place.val.llextra.is_some(), place.layout.is_unsized());

        if place.layout.is_zst() {
            return OperandRef::zero_sized(place.layout);
        }

        #[instrument(level = "trace", skip(bx))]
        fn scalar_load_metadata<'a, 'll, 'tcx>(
            bx: &mut Builder<'a, 'll, 'tcx>,
            load: &'ll Value,
            scalar: abi::Scalar,
            layout: TyAndLayout<'tcx>,
            offset: Size,
        ) {
            if bx.cx.sess().opts.optimize == OptLevel::No {
                // Don't emit metadata we're not going to use
                return;
            }

            if !scalar.is_uninit_valid() {
                bx.noundef_metadata(load);
            }

            match scalar.primitive() {
                abi::Primitive::Int(..) => {
                    if !scalar.is_always_valid(bx) {
                        bx.range_metadata(load, scalar.valid_range(bx));
                    }
                }
                abi::Primitive::Pointer(_) => {
                    if !scalar.valid_range(bx).contains(0) {
                        bx.nonnull_metadata(load);
                    }

                    if let Some(pointee) = layout.pointee_info_at(bx, offset) {
                        if let Some(_) = pointee.safe {
                            bx.align_metadata(load, pointee.align);
                        }
                    }
                }
                abi::Primitive::Float(_) => {}
            }
        }

        let val = if let Some(_) = place.val.llextra {
            // FIXME: Merge with the `else` below?
            OperandValue::Ref(place.val)
        } else if place.layout.is_llvm_immediate() {
            let mut const_llval = None;
            let llty = place.layout.llvm_type(self);
            unsafe {
                if let Some(global) = llvm::LLVMIsAGlobalVariable(place.val.llval) {
                    if llvm::LLVMIsGlobalConstant(global) == llvm::True {
                        if let Some(init) = llvm::LLVMGetInitializer(global) {
                            if self.val_ty(init) == llty {
                                const_llval = Some(init);
                            }
                        }
                    }
                }
            }
            let llval = const_llval.unwrap_or_else(|| {
                let load = self.load(llty, place.val.llval, place.val.align);
                if let abi::BackendRepr::Scalar(scalar) = place.layout.backend_repr {
                    scalar_load_metadata(self, load, scalar, place.layout, Size::ZERO);
                }
                load
            });
            OperandValue::Immediate(self.to_immediate(llval, place.layout))
        } else if let abi::BackendRepr::ScalarPair(a, b) = place.layout.backend_repr {
            let b_offset = a.size(self).align_to(b.align(self).abi);

            let mut load = |i, scalar: abi::Scalar, layout, align, offset| {
                let llptr = if i == 0 {
                    place.val.llval
                } else {
                    self.inbounds_ptradd(place.val.llval, self.const_usize(b_offset.bytes()))
                };
                let llty = place.layout.scalar_pair_element_llvm_type(self, i, false);
                let load = self.load(llty, llptr, align);
                scalar_load_metadata(self, load, scalar, layout, offset);
                self.to_immediate_scalar(load, scalar)
            };

            OperandValue::Pair(
                load(0, a, place.layout, place.val.align, Size::ZERO),
                load(1, b, place.layout, place.val.align.restrict_for_offset(b_offset), b_offset),
            )
        } else {
            OperandValue::Ref(place.val)
        };

        OperandRef { val, layout: place.layout }
    }

    fn write_operand_repeatedly(
        &mut self,
        cg_elem: OperandRef<'tcx, &'ll Value>,
        count: u64,
        dest: PlaceRef<'tcx, &'ll Value>,
    ) {
        let zero = self.const_usize(0);
        let count = self.const_usize(count);

        let header_bb = self.append_sibling_block("repeat_loop_header");
        let body_bb = self.append_sibling_block("repeat_loop_body");
        let next_bb = self.append_sibling_block("repeat_loop_next");

        self.br(header_bb);

        let mut header_bx = Self::build(self.cx, header_bb);
        let i = header_bx.phi(self.val_ty(zero), &[zero], &[self.llbb()]);

        let keep_going = header_bx.icmp(IntPredicate::IntULT, i, count);
        header_bx.cond_br(keep_going, body_bb, next_bb);

        let mut body_bx = Self::build(self.cx, body_bb);
        let dest_elem = dest.project_index(&mut body_bx, i);
        cg_elem.val.store(&mut body_bx, dest_elem);

        let next = body_bx.unchecked_uadd(i, self.const_usize(1));
        body_bx.br(header_bb);
        header_bx.add_incoming_to_phi(i, next, body_bb);

        *self = Self::build(self.cx, next_bb);
    }

    fn range_metadata(&mut self, load: &'ll Value, range: WrappingRange) {
        if self.sess().target.arch == "amdgpu" {
            // amdgpu/LLVM does something weird and thinks an i64 value is
            // split into a v2i32, halving the bitwidth LLVM expects,
            // tripping an assertion. So, for now, just disable this
            // optimization.
            return;
        }

        if self.cx.sess().opts.optimize == OptLevel::No {
            // Don't emit metadata we're not going to use
            return;
        }

        unsafe {
            let llty = self.cx.val_ty(load);
            let md = [
                llvm::LLVMValueAsMetadata(self.cx.const_uint_big(llty, range.start)),
                llvm::LLVMValueAsMetadata(self.cx.const_uint_big(llty, range.end.wrapping_add(1))),
            ];
            let md = llvm::LLVMMDNodeInContext2(self.cx.llcx, md.as_ptr(), md.len());
            self.set_metadata(load, llvm::MD_range, md);
        }
    }

    fn nonnull_metadata(&mut self, load: &'ll Value) {
        unsafe {
            let md = llvm::LLVMMDNodeInContext2(self.cx.llcx, ptr::null(), 0);
            self.set_metadata(load, llvm::MD_nonnull, md);
        }
    }

    fn store(&mut self, val: &'ll Value, ptr: &'ll Value, align: Align) -> &'ll Value {
        self.store_with_flags(val, ptr, align, MemFlags::empty())
    }

    fn store_with_flags(
        &mut self,
        val: &'ll Value,
        ptr: &'ll Value,
        align: Align,
        flags: MemFlags,
    ) -> &'ll Value {
        debug!("Store {:?} -> {:?} ({:?})", val, ptr, flags);
        assert_eq!(self.cx.type_kind(self.cx.val_ty(ptr)), TypeKind::Pointer);
        unsafe {
            let store = llvm::LLVMBuildStore(self.llbuilder, val, ptr);
            let align =
                if flags.contains(MemFlags::UNALIGNED) { 1 } else { align.bytes() as c_uint };
            llvm::LLVMSetAlignment(store, align);
            if flags.contains(MemFlags::VOLATILE) {
                llvm::LLVMSetVolatile(store, llvm::True);
            }
            if flags.contains(MemFlags::NONTEMPORAL) {
                // Make sure that the current target architectures supports "sane" non-temporal
                // stores, i.e., non-temporal stores that are equivalent to regular stores except
                // for performance. LLVM doesn't seem to care about this, and will happily treat
                // `!nontemporal` stores as-if they were normal stores (for reordering optimizations
                // etc) even on x86, despite later lowering them to MOVNT which do *not* behave like
                // regular stores but require special fences. So we keep a list of architectures
                // where `!nontemporal` is known to be truly just a hint, and use regular stores
                // everywhere else. (In the future, we could alternatively ensure that an sfence
                // gets emitted after a sequence of movnt before any kind of synchronizing
                // operation. But it's not clear how to do that with LLVM.)
                // For more context, see <https://github.com/rust-lang/rust/issues/114582> and
                // <https://github.com/llvm/llvm-project/issues/64521>.
                const WELL_BEHAVED_NONTEMPORAL_ARCHS: &[&str] =
                    &["aarch64", "arm", "riscv32", "riscv64"];

                let use_nontemporal =
                    WELL_BEHAVED_NONTEMPORAL_ARCHS.contains(&&*self.cx.tcx.sess.target.arch);
                if use_nontemporal {
                    // According to LLVM [1] building a nontemporal store must
                    // *always* point to a metadata value of the integer 1.
                    //
                    // [1]: https://llvm.org/docs/LangRef.html#store-instruction
                    let one = llvm::LLVMValueAsMetadata(self.cx.const_i32(1));
                    let md = llvm::LLVMMDNodeInContext2(self.cx.llcx, &one, 1);
                    self.set_metadata(store, llvm::MD_nontemporal, md);
                }
            }
            store
        }
    }

    fn atomic_store(
        &mut self,
        val: &'ll Value,
        ptr: &'ll Value,
        order: rustc_codegen_ssa::common::AtomicOrdering,
        size: Size,
    ) {
        debug!("Store {:?} -> {:?}", val, ptr);
        assert_eq!(self.cx.type_kind(self.cx.val_ty(ptr)), TypeKind::Pointer);
        unsafe {
            let store = llvm::LLVMRustBuildAtomicStore(
                self.llbuilder,
                val,
                ptr,
                AtomicOrdering::from_generic(order),
            );
            // LLVM requires the alignment of atomic stores to be at least the size of the type.
            llvm::LLVMSetAlignment(store, size.bytes() as c_uint);
        }
    }

    fn gep(&mut self, ty: &'ll Type, ptr: &'ll Value, indices: &[&'ll Value]) -> &'ll Value {
        unsafe {
            llvm::LLVMBuildGEP2(
                self.llbuilder,
                ty,
                ptr,
                indices.as_ptr(),
                indices.len() as c_uint,
                UNNAMED,
            )
        }
    }

    fn inbounds_gep(
        &mut self,
        ty: &'ll Type,
        ptr: &'ll Value,
        indices: &[&'ll Value],
    ) -> &'ll Value {
        unsafe {
            llvm::LLVMBuildInBoundsGEP2(
                self.llbuilder,
                ty,
                ptr,
                indices.as_ptr(),
                indices.len() as c_uint,
                UNNAMED,
            )
        }
    }

    /* Casts */
    fn trunc(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildTrunc(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn sext(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildSExt(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn fptoui_sat(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        self.fptoint_sat(false, val, dest_ty)
    }

    fn fptosi_sat(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        self.fptoint_sat(true, val, dest_ty)
    }

    fn fptoui(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        // On WebAssembly the `fptoui` and `fptosi` instructions currently have
        // poor codegen. The reason for this is that the corresponding wasm
        // instructions, `i32.trunc_f32_s` for example, will trap when the float
        // is out-of-bounds, infinity, or nan. This means that LLVM
        // automatically inserts control flow around `fptoui` and `fptosi`
        // because the LLVM instruction `fptoui` is defined as producing a
        // poison value, not having UB on out-of-bounds values.
        //
        // This method, however, is only used with non-saturating casts that
        // have UB on out-of-bounds values. This means that it's ok if we use
        // the raw wasm instruction since out-of-bounds values can do whatever
        // we like. To ensure that LLVM picks the right instruction we choose
        // the raw wasm intrinsic functions which avoid LLVM inserting all the
        // other control flow automatically.
        if self.sess().target.is_like_wasm {
            let src_ty = self.cx.val_ty(val);
            if self.cx.type_kind(src_ty) != TypeKind::Vector {
                let float_width = self.cx.float_width(src_ty);
                let int_width = self.cx.int_width(dest_ty);
                let name = match (int_width, float_width) {
                    (32, 32) => Some("llvm.wasm.trunc.unsigned.i32.f32"),
                    (32, 64) => Some("llvm.wasm.trunc.unsigned.i32.f64"),
                    (64, 32) => Some("llvm.wasm.trunc.unsigned.i64.f32"),
                    (64, 64) => Some("llvm.wasm.trunc.unsigned.i64.f64"),
                    _ => None,
                };
                if let Some(name) = name {
                    return self.call_intrinsic(name, &[val]);
                }
            }
        }
        unsafe { llvm::LLVMBuildFPToUI(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn fptosi(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        // see `fptoui` above for why wasm is different here
        if self.sess().target.is_like_wasm {
            let src_ty = self.cx.val_ty(val);
            if self.cx.type_kind(src_ty) != TypeKind::Vector {
                let float_width = self.cx.float_width(src_ty);
                let int_width = self.cx.int_width(dest_ty);
                let name = match (int_width, float_width) {
                    (32, 32) => Some("llvm.wasm.trunc.signed.i32.f32"),
                    (32, 64) => Some("llvm.wasm.trunc.signed.i32.f64"),
                    (64, 32) => Some("llvm.wasm.trunc.signed.i64.f32"),
                    (64, 64) => Some("llvm.wasm.trunc.signed.i64.f64"),
                    _ => None,
                };
                if let Some(name) = name {
                    return self.call_intrinsic(name, &[val]);
                }
            }
        }
        unsafe { llvm::LLVMBuildFPToSI(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn uitofp(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildUIToFP(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn sitofp(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildSIToFP(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn fptrunc(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildFPTrunc(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn fpext(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildFPExt(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn ptrtoint(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildPtrToInt(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn inttoptr(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildIntToPtr(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn bitcast(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildBitCast(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn intcast(&mut self, val: &'ll Value, dest_ty: &'ll Type, is_signed: bool) -> &'ll Value {
        unsafe {
            llvm::LLVMBuildIntCast2(
                self.llbuilder,
                val,
                dest_ty,
                if is_signed { True } else { False },
                UNNAMED,
            )
        }
    }

    fn pointercast(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildPointerCast(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    /* Comparisons */
    fn icmp(&mut self, op: IntPredicate, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        let op = llvm::IntPredicate::from_generic(op);
        unsafe { llvm::LLVMBuildICmp(self.llbuilder, op as c_uint, lhs, rhs, UNNAMED) }
    }

    fn fcmp(&mut self, op: RealPredicate, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        let op = llvm::RealPredicate::from_generic(op);
        unsafe { llvm::LLVMBuildFCmp(self.llbuilder, op as c_uint, lhs, rhs, UNNAMED) }
    }

    /* Miscellaneous instructions */
    fn memcpy(
        &mut self,
        dst: &'ll Value,
        dst_align: Align,
        src: &'ll Value,
        src_align: Align,
        size: &'ll Value,
        flags: MemFlags,
    ) {
        assert!(!flags.contains(MemFlags::NONTEMPORAL), "non-temporal memcpy not supported");
        let size = self.intcast(size, self.type_isize(), false);
        let is_volatile = flags.contains(MemFlags::VOLATILE);
        unsafe {
            llvm::LLVMRustBuildMemCpy(
                self.llbuilder,
                dst,
                dst_align.bytes() as c_uint,
                src,
                src_align.bytes() as c_uint,
                size,
                is_volatile,
            );
        }
    }

    fn memmove(
        &mut self,
        dst: &'ll Value,
        dst_align: Align,
        src: &'ll Value,
        src_align: Align,
        size: &'ll Value,
        flags: MemFlags,
    ) {
        assert!(!flags.contains(MemFlags::NONTEMPORAL), "non-temporal memmove not supported");
        let size = self.intcast(size, self.type_isize(), false);
        let is_volatile = flags.contains(MemFlags::VOLATILE);
        unsafe {
            llvm::LLVMRustBuildMemMove(
                self.llbuilder,
                dst,
                dst_align.bytes() as c_uint,
                src,
                src_align.bytes() as c_uint,
                size,
                is_volatile,
            );
        }
    }

    fn memset(
        &mut self,
        ptr: &'ll Value,
        fill_byte: &'ll Value,
        size: &'ll Value,
        align: Align,
        flags: MemFlags,
    ) {
        assert!(!flags.contains(MemFlags::NONTEMPORAL), "non-temporal memset not supported");
        let is_volatile = flags.contains(MemFlags::VOLATILE);
        unsafe {
            llvm::LLVMRustBuildMemSet(
                self.llbuilder,
                ptr,
                align.bytes() as c_uint,
                fill_byte,
                size,
                is_volatile,
            );
        }
    }

    fn select(
        &mut self,
        cond: &'ll Value,
        then_val: &'ll Value,
        else_val: &'ll Value,
    ) -> &'ll Value {
        unsafe { llvm::LLVMBuildSelect(self.llbuilder, cond, then_val, else_val, UNNAMED) }
    }

    fn va_arg(&mut self, list: &'ll Value, ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildVAArg(self.llbuilder, list, ty, UNNAMED) }
    }

    fn extract_element(&mut self, vec: &'ll Value, idx: &'ll Value) -> &'ll Value {
        unsafe { llvm::LLVMBuildExtractElement(self.llbuilder, vec, idx, UNNAMED) }
    }

    fn vector_splat(&mut self, num_elts: usize, elt: &'ll Value) -> &'ll Value {
        unsafe {
            let elt_ty = self.cx.val_ty(elt);
            let undef = llvm::LLVMGetUndef(self.type_vector(elt_ty, num_elts as u64));
            let vec = self.insert_element(undef, elt, self.cx.const_i32(0));
            let vec_i32_ty = self.type_vector(self.type_i32(), num_elts as u64);
            self.shuffle_vector(vec, undef, self.const_null(vec_i32_ty))
        }
    }

    fn extract_value(&mut self, agg_val: &'ll Value, idx: u64) -> &'ll Value {
        assert_eq!(idx as c_uint as u64, idx);
        unsafe { llvm::LLVMBuildExtractValue(self.llbuilder, agg_val, idx as c_uint, UNNAMED) }
    }

    fn insert_value(&mut self, agg_val: &'ll Value, elt: &'ll Value, idx: u64) -> &'ll Value {
        assert_eq!(idx as c_uint as u64, idx);
        unsafe { llvm::LLVMBuildInsertValue(self.llbuilder, agg_val, elt, idx as c_uint, UNNAMED) }
    }

    fn set_personality_fn(&mut self, personality: &'ll Value) {
        unsafe {
            llvm::LLVMSetPersonalityFn(self.llfn(), personality);
        }
    }

    fn cleanup_landing_pad(&mut self, pers_fn: &'ll Value) -> (&'ll Value, &'ll Value) {
        let ty = self.type_struct(&[self.type_ptr(), self.type_i32()], false);
        let landing_pad = self.landing_pad(ty, pers_fn, 0);
        unsafe {
            llvm::LLVMSetCleanup(landing_pad, llvm::True);
        }
        (self.extract_value(landing_pad, 0), self.extract_value(landing_pad, 1))
    }

    fn filter_landing_pad(&mut self, pers_fn: &'ll Value) -> (&'ll Value, &'ll Value) {
        let ty = self.type_struct(&[self.type_ptr(), self.type_i32()], false);
        let landing_pad = self.landing_pad(ty, pers_fn, 1);
        self.add_clause(landing_pad, self.const_array(self.type_ptr(), &[]));
        (self.extract_value(landing_pad, 0), self.extract_value(landing_pad, 1))
    }

    fn resume(&mut self, exn0: &'ll Value, exn1: &'ll Value) {
        let ty = self.type_struct(&[self.type_ptr(), self.type_i32()], false);
        let mut exn = self.const_poison(ty);
        exn = self.insert_value(exn, exn0, 0);
        exn = self.insert_value(exn, exn1, 1);
        unsafe {
            llvm::LLVMBuildResume(self.llbuilder, exn);
        }
    }

    fn cleanup_pad(&mut self, parent: Option<&'ll Value>, args: &[&'ll Value]) -> Funclet<'ll> {
        let ret = unsafe {
            llvm::LLVMBuildCleanupPad(
                self.llbuilder,
                parent,
                args.as_ptr(),
                args.len() as c_uint,
                c"cleanuppad".as_ptr(),
            )
        };
        Funclet::new(ret.expect("LLVM does not have support for cleanuppad"))
    }

    fn cleanup_ret(&mut self, funclet: &Funclet<'ll>, unwind: Option<&'ll BasicBlock>) {
        unsafe {
            llvm::LLVMBuildCleanupRet(self.llbuilder, funclet.cleanuppad(), unwind)
                .expect("LLVM does not have support for cleanupret");
        }
    }

    fn catch_pad(&mut self, parent: &'ll Value, args: &[&'ll Value]) -> Funclet<'ll> {
        let ret = unsafe {
            llvm::LLVMBuildCatchPad(
                self.llbuilder,
                parent,
                args.as_ptr(),
                args.len() as c_uint,
                c"catchpad".as_ptr(),
            )
        };
        Funclet::new(ret.expect("LLVM does not have support for catchpad"))
    }

    fn catch_switch(
        &mut self,
        parent: Option<&'ll Value>,
        unwind: Option<&'ll BasicBlock>,
        handlers: &[&'ll BasicBlock],
    ) -> &'ll Value {
        let ret = unsafe {
            llvm::LLVMBuildCatchSwitch(
                self.llbuilder,
                parent,
                unwind,
                handlers.len() as c_uint,
                c"catchswitch".as_ptr(),
            )
        };
        let ret = ret.expect("LLVM does not have support for catchswitch");
        for handler in handlers {
            unsafe {
                llvm::LLVMAddHandler(ret, handler);
            }
        }
        ret
    }

    // Atomic Operations
    fn atomic_cmpxchg(
        &mut self,
        dst: &'ll Value,
        cmp: &'ll Value,
        src: &'ll Value,
        order: rustc_codegen_ssa::common::AtomicOrdering,
        failure_order: rustc_codegen_ssa::common::AtomicOrdering,
        weak: bool,
    ) -> (&'ll Value, &'ll Value) {
        let weak = if weak { llvm::True } else { llvm::False };
        unsafe {
            let value = llvm::LLVMBuildAtomicCmpXchg(
                self.llbuilder,
                dst,
                cmp,
                src,
                AtomicOrdering::from_generic(order),
                AtomicOrdering::from_generic(failure_order),
                llvm::False, // SingleThreaded
            );
            llvm::LLVMSetWeak(value, weak);
            let val = self.extract_value(value, 0);
            let success = self.extract_value(value, 1);
            (val, success)
        }
    }

    fn atomic_rmw(
        &mut self,
        op: rustc_codegen_ssa::common::AtomicRmwBinOp,
        dst: &'ll Value,
        mut src: &'ll Value,
        order: rustc_codegen_ssa::common::AtomicOrdering,
    ) -> &'ll Value {
        // The only RMW operation that LLVM supports on pointers is compare-exchange.
        let requires_cast_to_int = self.val_ty(src) == self.type_ptr()
            && op != rustc_codegen_ssa::common::AtomicRmwBinOp::AtomicXchg;
        if requires_cast_to_int {
            src = self.ptrtoint(src, self.type_isize());
        }
        let mut res = unsafe {
            llvm::LLVMBuildAtomicRMW(
                self.llbuilder,
                AtomicRmwBinOp::from_generic(op),
                dst,
                src,
                AtomicOrdering::from_generic(order),
                llvm::False, // SingleThreaded
            )
        };
        if requires_cast_to_int {
            res = self.inttoptr(res, self.type_ptr());
        }
        res
    }

    fn atomic_fence(
        &mut self,
        order: rustc_codegen_ssa::common::AtomicOrdering,
        scope: SynchronizationScope,
    ) {
        let single_threaded = match scope {
            SynchronizationScope::SingleThread => llvm::True,
            SynchronizationScope::CrossThread => llvm::False,
        };
        unsafe {
            llvm::LLVMBuildFence(
                self.llbuilder,
                AtomicOrdering::from_generic(order),
                single_threaded,
                UNNAMED,
            );
        }
    }

    fn set_invariant_load(&mut self, load: &'ll Value) {
        unsafe {
            let md = llvm::LLVMMDNodeInContext2(self.cx.llcx, ptr::null(), 0);
            self.set_metadata(load, llvm::MD_invariant_load, md);
        }
    }

    fn lifetime_start(&mut self, ptr: &'ll Value, size: Size) {
        self.call_lifetime_intrinsic("llvm.lifetime.start.p0i8", ptr, size);
    }

    fn lifetime_end(&mut self, ptr: &'ll Value, size: Size) {
        self.call_lifetime_intrinsic("llvm.lifetime.end.p0i8", ptr, size);
    }

    fn call(
        &mut self,
        llty: &'ll Type,
        fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        llfn: &'ll Value,
        args: &[&'ll Value],
        funclet: Option<&Funclet<'ll>>,
        instance: Option<Instance<'tcx>>,
    ) -> &'ll Value {
        debug!("call {:?} with args ({:?})", llfn, args);

        let args = self.check_call("call", llty, llfn, args);
        let funclet_bundle = funclet.map(|funclet| funclet.bundle());
        let mut bundles: SmallVec<[_; 2]> = SmallVec::new();
        if let Some(funclet_bundle) = funclet_bundle {
            bundles.push(funclet_bundle);
        }

        // Emit CFI pointer type membership test
        self.cfi_type_test(fn_attrs, fn_abi, instance, llfn);

        // Emit KCFI operand bundle
        let kcfi_bundle = self.kcfi_operand_bundle(fn_attrs, fn_abi, instance, llfn);
        if let Some(kcfi_bundle) = kcfi_bundle.as_deref() {
            bundles.push(kcfi_bundle);
        }

        let call = unsafe {
            llvm::LLVMBuildCallWithOperandBundles(
                self.llbuilder,
                llty,
                llfn,
                args.as_ptr() as *const &llvm::Value,
                args.len() as c_uint,
                bundles.as_ptr(),
                bundles.len() as c_uint,
                c"".as_ptr(),
            )
        };
        if let Some(fn_abi) = fn_abi {
            fn_abi.apply_attrs_callsite(self, call);
        }
        call
    }

    fn zext(&mut self, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildZExt(self.llbuilder, val, dest_ty, UNNAMED) }
    }

    fn apply_attrs_to_cleanup_callsite(&mut self, llret: &'ll Value) {
        // Cleanup is always the cold path.
        let cold_inline = llvm::AttributeKind::Cold.create_attr(self.llcx);
        attributes::apply_to_callsite(llret, llvm::AttributePlace::Function, &[cold_inline]);
    }
}

impl<'ll> StaticBuilderMethods for Builder<'_, 'll, '_> {
    fn get_static(&mut self, def_id: DefId) -> &'ll Value {
        // Forward to the `get_static` method of `CodegenCx`
        self.cx().get_static(def_id)
    }
}

impl<'a, 'll, 'tcx> Builder<'a, 'll, 'tcx> {
    fn with_cx(cx: &'a CodegenCx<'ll, 'tcx>) -> Self {
        // Create a fresh builder from the crate context.
        let llbuilder = unsafe { llvm::LLVMCreateBuilderInContext(cx.llcx) };
        Builder { llbuilder, cx }
    }

    pub(crate) fn llfn(&self) -> &'ll Value {
        unsafe { llvm::LLVMGetBasicBlockParent(self.llbb()) }
    }

    fn position_at_start(&mut self, llbb: &'ll BasicBlock) {
        unsafe {
            llvm::LLVMRustPositionBuilderAtStart(self.llbuilder, llbb);
        }
    }

    fn align_metadata(&mut self, load: &'ll Value, align: Align) {
        unsafe {
            let md = [llvm::LLVMValueAsMetadata(self.cx.const_u64(align.bytes()))];
            let md = llvm::LLVMMDNodeInContext2(self.cx.llcx, md.as_ptr(), md.len());
            self.set_metadata(load, llvm::MD_align, md);
        }
    }

    fn noundef_metadata(&mut self, load: &'ll Value) {
        unsafe {
            let md = llvm::LLVMMDNodeInContext2(self.cx.llcx, ptr::null(), 0);
            self.set_metadata(load, llvm::MD_noundef, md);
        }
    }

    pub(crate) fn set_unpredictable(&mut self, inst: &'ll Value) {
        unsafe {
            let md = llvm::LLVMMDNodeInContext2(self.cx.llcx, ptr::null(), 0);
            self.set_metadata(inst, llvm::MD_unpredictable, md);
        }
    }

    pub(crate) fn minnum(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        unsafe { llvm::LLVMRustBuildMinNum(self.llbuilder, lhs, rhs) }
    }

    pub(crate) fn maxnum(&mut self, lhs: &'ll Value, rhs: &'ll Value) -> &'ll Value {
        unsafe { llvm::LLVMRustBuildMaxNum(self.llbuilder, lhs, rhs) }
    }

    pub(crate) fn insert_element(
        &mut self,
        vec: &'ll Value,
        elt: &'ll Value,
        idx: &'ll Value,
    ) -> &'ll Value {
        unsafe { llvm::LLVMBuildInsertElement(self.llbuilder, vec, elt, idx, UNNAMED) }
    }

    pub(crate) fn shuffle_vector(
        &mut self,
        v1: &'ll Value,
        v2: &'ll Value,
        mask: &'ll Value,
    ) -> &'ll Value {
        unsafe { llvm::LLVMBuildShuffleVector(self.llbuilder, v1, v2, mask, UNNAMED) }
    }

    pub(crate) fn vector_reduce_fadd(&mut self, acc: &'ll Value, src: &'ll Value) -> &'ll Value {
        unsafe { llvm::LLVMRustBuildVectorReduceFAdd(self.llbuilder, acc, src) }
    }
    pub(crate) fn vector_reduce_fmul(&mut self, acc: &'ll Value, src: &'ll Value) -> &'ll Value {
        unsafe { llvm::LLVMRustBuildVectorReduceFMul(self.llbuilder, acc, src) }
    }
    pub(crate) fn vector_reduce_fadd_reassoc(
        &mut self,
        acc: &'ll Value,
        src: &'ll Value,
    ) -> &'ll Value {
        unsafe {
            let instr = llvm::LLVMRustBuildVectorReduceFAdd(self.llbuilder, acc, src);
            llvm::LLVMRustSetAllowReassoc(instr);
            instr
        }
    }
    pub(crate) fn vector_reduce_fmul_reassoc(
        &mut self,
        acc: &'ll Value,
        src: &'ll Value,
    ) -> &'ll Value {
        unsafe {
            let instr = llvm::LLVMRustBuildVectorReduceFMul(self.llbuilder, acc, src);
            llvm::LLVMRustSetAllowReassoc(instr);
            instr
        }
    }
    pub(crate) fn vector_reduce_add(&mut self, src: &'ll Value) -> &'ll Value {
        unsafe { llvm::LLVMRustBuildVectorReduceAdd(self.llbuilder, src) }
    }
    pub(crate) fn vector_reduce_mul(&mut self, src: &'ll Value) -> &'ll Value {
        unsafe { llvm::LLVMRustBuildVectorReduceMul(self.llbuilder, src) }
    }
    pub(crate) fn vector_reduce_and(&mut self, src: &'ll Value) -> &'ll Value {
        unsafe { llvm::LLVMRustBuildVectorReduceAnd(self.llbuilder, src) }
    }
    pub(crate) fn vector_reduce_or(&mut self, src: &'ll Value) -> &'ll Value {
        unsafe { llvm::LLVMRustBuildVectorReduceOr(self.llbuilder, src) }
    }
    pub(crate) fn vector_reduce_xor(&mut self, src: &'ll Value) -> &'ll Value {
        unsafe { llvm::LLVMRustBuildVectorReduceXor(self.llbuilder, src) }
    }
    pub(crate) fn vector_reduce_fmin(&mut self, src: &'ll Value) -> &'ll Value {
        unsafe {
            llvm::LLVMRustBuildVectorReduceFMin(self.llbuilder, src, /*NoNaNs:*/ false)
        }
    }
    pub(crate) fn vector_reduce_fmax(&mut self, src: &'ll Value) -> &'ll Value {
        unsafe {
            llvm::LLVMRustBuildVectorReduceFMax(self.llbuilder, src, /*NoNaNs:*/ false)
        }
    }
    pub(crate) fn vector_reduce_min(&mut self, src: &'ll Value, is_signed: bool) -> &'ll Value {
        unsafe { llvm::LLVMRustBuildVectorReduceMin(self.llbuilder, src, is_signed) }
    }
    pub(crate) fn vector_reduce_max(&mut self, src: &'ll Value, is_signed: bool) -> &'ll Value {
        unsafe { llvm::LLVMRustBuildVectorReduceMax(self.llbuilder, src, is_signed) }
    }

    pub(crate) fn add_clause(&mut self, landing_pad: &'ll Value, clause: &'ll Value) {
        unsafe {
            llvm::LLVMAddClause(landing_pad, clause);
        }
    }

    pub(crate) fn catch_ret(
        &mut self,
        funclet: &Funclet<'ll>,
        unwind: &'ll BasicBlock,
    ) -> &'ll Value {
        let ret = unsafe { llvm::LLVMBuildCatchRet(self.llbuilder, funclet.cleanuppad(), unwind) };
        ret.expect("LLVM does not have support for catchret")
    }

    fn check_call<'b>(
        &mut self,
        typ: &str,
        fn_ty: &'ll Type,
        llfn: &'ll Value,
        args: &'b [&'ll Value],
    ) -> Cow<'b, [&'ll Value]> {
        assert!(
            self.cx.type_kind(fn_ty) == TypeKind::Function,
            "builder::{typ} not passed a function, but {fn_ty:?}"
        );

        let param_tys = self.cx.func_params_types(fn_ty);

        let all_args_match = iter::zip(&param_tys, args.iter().map(|&v| self.val_ty(v)))
            .all(|(expected_ty, actual_ty)| *expected_ty == actual_ty);

        if all_args_match {
            return Cow::Borrowed(args);
        }

        let casted_args: Vec<_> = iter::zip(param_tys, args)
            .enumerate()
            .map(|(i, (expected_ty, &actual_val))| {
                let actual_ty = self.val_ty(actual_val);
                if expected_ty != actual_ty {
                    debug!(
                        "type mismatch in function call of {:?}. \
                            Expected {:?} for param {}, got {:?}; injecting bitcast",
                        llfn, expected_ty, i, actual_ty
                    );
                    self.bitcast(actual_val, expected_ty)
                } else {
                    actual_val
                }
            })
            .collect();

        Cow::Owned(casted_args)
    }

    pub(crate) fn va_arg(&mut self, list: &'ll Value, ty: &'ll Type) -> &'ll Value {
        unsafe { llvm::LLVMBuildVAArg(self.llbuilder, list, ty, UNNAMED) }
    }

    pub(crate) fn call_intrinsic(&mut self, intrinsic: &str, args: &[&'ll Value]) -> &'ll Value {
        let (ty, f) = self.cx.get_intrinsic(intrinsic);
        self.call(ty, None, None, f, args, None, None)
    }

    fn call_lifetime_intrinsic(&mut self, intrinsic: &str, ptr: &'ll Value, size: Size) {
        let size = size.bytes();
        if size == 0 {
            return;
        }

        if !self.cx().sess().emit_lifetime_markers() {
            return;
        }

        self.call_intrinsic(intrinsic, &[self.cx.const_u64(size), ptr]);
    }

    pub(crate) fn phi(
        &mut self,
        ty: &'ll Type,
        vals: &[&'ll Value],
        bbs: &[&'ll BasicBlock],
    ) -> &'ll Value {
        assert_eq!(vals.len(), bbs.len());
        let phi = unsafe { llvm::LLVMBuildPhi(self.llbuilder, ty, UNNAMED) };
        unsafe {
            llvm::LLVMAddIncoming(phi, vals.as_ptr(), bbs.as_ptr(), vals.len() as c_uint);
            phi
        }
    }

    fn add_incoming_to_phi(&mut self, phi: &'ll Value, val: &'ll Value, bb: &'ll BasicBlock) {
        unsafe {
            llvm::LLVMAddIncoming(phi, &val, &bb, 1 as c_uint);
        }
    }

    fn fptoint_sat(&mut self, signed: bool, val: &'ll Value, dest_ty: &'ll Type) -> &'ll Value {
        let src_ty = self.cx.val_ty(val);
        let (float_ty, int_ty, vector_length) = if self.cx.type_kind(src_ty) == TypeKind::Vector {
            assert_eq!(self.cx.vector_length(src_ty), self.cx.vector_length(dest_ty));
            (
                self.cx.element_type(src_ty),
                self.cx.element_type(dest_ty),
                Some(self.cx.vector_length(src_ty)),
            )
        } else {
            (src_ty, dest_ty, None)
        };
        let float_width = self.cx.float_width(float_ty);
        let int_width = self.cx.int_width(int_ty);

        let instr = if signed { "fptosi" } else { "fptoui" };
        let name = if let Some(vector_length) = vector_length {
            format!("llvm.{instr}.sat.v{vector_length}i{int_width}.v{vector_length}f{float_width}")
        } else {
            format!("llvm.{instr}.sat.i{int_width}.f{float_width}")
        };
        let f = self.declare_cfn(&name, llvm::UnnamedAddr::No, self.type_func(&[src_ty], dest_ty));
        self.call(self.type_func(&[src_ty], dest_ty), None, None, f, &[val], None, None)
    }

    pub(crate) fn landing_pad(
        &mut self,
        ty: &'ll Type,
        pers_fn: &'ll Value,
        num_clauses: usize,
    ) -> &'ll Value {
        // Use LLVMSetPersonalityFn to set the personality. It supports arbitrary Consts while,
        // LLVMBuildLandingPad requires the argument to be a Function (as of LLVM 12). The
        // personality lives on the parent function anyway.
        self.set_personality_fn(pers_fn);
        unsafe {
            llvm::LLVMBuildLandingPad(self.llbuilder, ty, None, num_clauses as c_uint, UNNAMED)
        }
    }

    pub(crate) fn callbr(
        &mut self,
        llty: &'ll Type,
        fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        llfn: &'ll Value,
        args: &[&'ll Value],
        default_dest: &'ll BasicBlock,
        indirect_dest: &[&'ll BasicBlock],
        funclet: Option<&Funclet<'ll>>,
        instance: Option<Instance<'tcx>>,
    ) -> &'ll Value {
        debug!("invoke {:?} with args ({:?})", llfn, args);

        let args = self.check_call("callbr", llty, llfn, args);
        let funclet_bundle = funclet.map(|funclet| funclet.bundle());
        let mut bundles: SmallVec<[_; 2]> = SmallVec::new();
        if let Some(funclet_bundle) = funclet_bundle {
            bundles.push(funclet_bundle);
        }

        // Emit CFI pointer type membership test
        self.cfi_type_test(fn_attrs, fn_abi, instance, llfn);

        // Emit KCFI operand bundle
        let kcfi_bundle = self.kcfi_operand_bundle(fn_attrs, fn_abi, instance, llfn);
        if let Some(kcfi_bundle) = kcfi_bundle.as_deref() {
            bundles.push(kcfi_bundle);
        }

        let callbr = unsafe {
            llvm::LLVMBuildCallBr(
                self.llbuilder,
                llty,
                llfn,
                default_dest,
                indirect_dest.as_ptr(),
                indirect_dest.len() as c_uint,
                args.as_ptr(),
                args.len() as c_uint,
                bundles.as_ptr(),
                bundles.len() as c_uint,
                UNNAMED,
            )
        };
        if let Some(fn_abi) = fn_abi {
            fn_abi.apply_attrs_callsite(self, callbr);
        }
        callbr
    }

    // Emits CFI pointer type membership tests.
    fn cfi_type_test(
        &mut self,
        fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        instance: Option<Instance<'tcx>>,
        llfn: &'ll Value,
    ) {
        let is_indirect_call = unsafe { llvm::LLVMRustIsNonGVFunctionPointerTy(llfn) };
        if self.tcx.sess.is_sanitizer_cfi_enabled()
            && let Some(fn_abi) = fn_abi
            && is_indirect_call
        {
            if let Some(fn_attrs) = fn_attrs
                && fn_attrs.no_sanitize.contains(SanitizerSet::CFI)
            {
                return;
            }

            let mut options = cfi::TypeIdOptions::empty();
            if self.tcx.sess.is_sanitizer_cfi_generalize_pointers_enabled() {
                options.insert(cfi::TypeIdOptions::GENERALIZE_POINTERS);
            }
            if self.tcx.sess.is_sanitizer_cfi_normalize_integers_enabled() {
                options.insert(cfi::TypeIdOptions::NORMALIZE_INTEGERS);
            }

            let typeid = if let Some(instance) = instance {
                cfi::typeid_for_instance(self.tcx, instance, options)
            } else {
                cfi::typeid_for_fnabi(self.tcx, fn_abi, options)
            };
            let typeid_metadata = self.cx.typeid_metadata(typeid).unwrap();
            let dbg_loc = self.get_dbg_loc();

            // Test whether the function pointer is associated with the type identifier.
            let cond = self.type_test(llfn, typeid_metadata);
            let bb_pass = self.append_sibling_block("type_test.pass");
            let bb_fail = self.append_sibling_block("type_test.fail");
            self.cond_br(cond, bb_pass, bb_fail);

            self.switch_to_block(bb_fail);
            if let Some(dbg_loc) = dbg_loc {
                self.set_dbg_loc(dbg_loc);
            }
            self.abort();
            self.unreachable();

            self.switch_to_block(bb_pass);
            if let Some(dbg_loc) = dbg_loc {
                self.set_dbg_loc(dbg_loc);
            }
        }
    }

    // Emits KCFI operand bundles.
    fn kcfi_operand_bundle(
        &mut self,
        fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        instance: Option<Instance<'tcx>>,
        llfn: &'ll Value,
    ) -> Option<llvm::OperandBundleOwned<'ll>> {
        let is_indirect_call = unsafe { llvm::LLVMRustIsNonGVFunctionPointerTy(llfn) };
        let kcfi_bundle = if self.tcx.sess.is_sanitizer_kcfi_enabled()
            && let Some(fn_abi) = fn_abi
            && is_indirect_call
        {
            if let Some(fn_attrs) = fn_attrs
                && fn_attrs.no_sanitize.contains(SanitizerSet::KCFI)
            {
                return None;
            }

            let mut options = kcfi::TypeIdOptions::empty();
            if self.tcx.sess.is_sanitizer_cfi_generalize_pointers_enabled() {
                options.insert(kcfi::TypeIdOptions::GENERALIZE_POINTERS);
            }
            if self.tcx.sess.is_sanitizer_cfi_normalize_integers_enabled() {
                options.insert(kcfi::TypeIdOptions::NORMALIZE_INTEGERS);
            }

            let kcfi_typeid = if let Some(instance) = instance {
                kcfi::typeid_for_instance(self.tcx, instance, options)
            } else {
                kcfi::typeid_for_fnabi(self.tcx, fn_abi, options)
            };

            Some(llvm::OperandBundleOwned::new("kcfi", &[self.const_u32(kcfi_typeid)]))
        } else {
            None
        };
        kcfi_bundle
    }

    /// Emits a call to `llvm.instrprof.increment`. Used by coverage instrumentation.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn instrprof_increment(
        &mut self,
        fn_name: &'ll Value,
        hash: &'ll Value,
        num_counters: &'ll Value,
        index: &'ll Value,
    ) {
        self.call_intrinsic("llvm.instrprof.increment", &[fn_name, hash, num_counters, index]);
    }

    /// Emits a call to `llvm.instrprof.mcdc.parameters`.
    ///
    /// This doesn't produce any code directly, but is used as input by
    /// the LLVM pass that handles coverage instrumentation.
    ///
    /// (See clang's [`CodeGenPGO::emitMCDCParameters`] for comparison.)
    ///
    /// [`CodeGenPGO::emitMCDCParameters`]:
    ///     https://github.com/rust-lang/llvm-project/blob/5399a24/clang/lib/CodeGen/CodeGenPGO.cpp#L1124
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn mcdc_parameters(
        &mut self,
        fn_name: &'ll Value,
        hash: &'ll Value,
        bitmap_bits: &'ll Value,
    ) {
        assert!(
            crate::llvm_util::get_version() >= (19, 0, 0),
            "MCDC intrinsics require LLVM 19 or later"
        );
        self.call_intrinsic("llvm.instrprof.mcdc.parameters", &[fn_name, hash, bitmap_bits]);
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn mcdc_tvbitmap_update(
        &mut self,
        fn_name: &'ll Value,
        hash: &'ll Value,
        bitmap_index: &'ll Value,
        mcdc_temp: &'ll Value,
    ) {
        assert!(
            crate::llvm_util::get_version() >= (19, 0, 0),
            "MCDC intrinsics require LLVM 19 or later"
        );
        let args = &[fn_name, hash, bitmap_index, mcdc_temp];
        self.call_intrinsic("llvm.instrprof.mcdc.tvbitmap.update", args);
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn mcdc_condbitmap_reset(&mut self, mcdc_temp: &'ll Value) {
        self.store(self.const_i32(0), mcdc_temp, self.tcx.data_layout.i32_align.abi);
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn mcdc_condbitmap_update(&mut self, cond_index: &'ll Value, mcdc_temp: &'ll Value) {
        assert!(
            crate::llvm_util::get_version() >= (19, 0, 0),
            "MCDC intrinsics require LLVM 19 or later"
        );
        let align = self.tcx.data_layout.i32_align.abi;
        let current_tv_index = self.load(self.cx.type_i32(), mcdc_temp, align);
        let new_tv_index = self.add(current_tv_index, cond_index);
        self.store(new_tv_index, mcdc_temp, align);
    }
}
