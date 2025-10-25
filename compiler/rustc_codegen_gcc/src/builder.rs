use std::borrow::Cow;
use std::cell::Cell;
use std::convert::TryFrom;
use std::ops::Deref;

use gccjit::{
    BinaryOp, Block, ComparisonOp, Context, Function, LValue, Location, RValue, ToRValue, Type,
    UnaryOp,
};
use rustc_abi as abi;
use rustc_abi::{Align, HasDataLayout, Size, TargetDataLayout, WrappingRange};
use rustc_apfloat::{Float, Round, Status, ieee};
use rustc_codegen_ssa::MemFlags;
use rustc_codegen_ssa::common::{
    AtomicRmwBinOp, IntPredicate, RealPredicate, SynchronizationScope, TypeKind,
};
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::{
    BackendTypes, BaseTypeCodegenMethods, BuilderMethods, ConstCodegenMethods,
    LayoutTypeCodegenMethods, OverflowOp, StaticBuilderMethods,
};
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::ty::layout::{
    FnAbiError, FnAbiOfHelpers, FnAbiRequest, HasTyCtxt, HasTypingEnv, LayoutError, LayoutOfHelpers,
};
use rustc_middle::ty::{self, AtomicOrdering, Instance, Ty, TyCtxt};
use rustc_span::Span;
use rustc_span::def_id::DefId;
use rustc_target::callconv::FnAbi;
use rustc_target::spec::{HasTargetSpec, HasX86AbiOpt, Target, X86Abi};

use crate::common::{SignType, TypeReflection, type_is_pointer};
use crate::context::CodegenCx;
use crate::errors;
use crate::intrinsic::llvm;
use crate::type_of::LayoutGccExt;

// TODO(antoyo)
type Funclet = ();

enum ExtremumOperation {
    Max,
    Min,
}

pub struct Builder<'a, 'gcc, 'tcx> {
    pub cx: &'a CodegenCx<'gcc, 'tcx>,
    pub block: Block<'gcc>,
    pub location: Option<Location<'gcc>>,
    value_counter: Cell<u64>,
}

impl<'a, 'gcc, 'tcx> Builder<'a, 'gcc, 'tcx> {
    fn with_cx(cx: &'a CodegenCx<'gcc, 'tcx>, block: Block<'gcc>) -> Self {
        Builder { cx, block, location: None, value_counter: Cell::new(0) }
    }

    fn next_value_counter(&self) -> u64 {
        self.value_counter.set(self.value_counter.get() + 1);
        self.value_counter.get()
    }

    fn atomic_extremum(
        &mut self,
        operation: ExtremumOperation,
        dst: RValue<'gcc>,
        src: RValue<'gcc>,
        order: AtomicOrdering,
    ) -> RValue<'gcc> {
        let size = get_maybe_pointer_size(src);

        let func = self.current_func();

        let load_ordering = match order {
            // TODO(antoyo): does this make sense?
            AtomicOrdering::AcqRel | AtomicOrdering::Release => AtomicOrdering::Acquire,
            _ => order,
        };
        let previous_value =
            self.atomic_load(dst.get_type(), dst, load_ordering, Size::from_bytes(size));
        let previous_var =
            func.new_local(self.location, previous_value.get_type(), "previous_value");
        let return_value = func.new_local(self.location, previous_value.get_type(), "return_value");
        self.llbb().add_assignment(self.location, previous_var, previous_value);
        self.llbb().add_assignment(self.location, return_value, previous_var.to_rvalue());

        let while_block = func.new_block("while");
        let after_block = func.new_block("after_while");
        self.llbb().end_with_jump(self.location, while_block);

        // NOTE: since jumps were added and compare_exchange doesn't expect this, the current block in the
        // state need to be updated.
        self.switch_to_block(while_block);

        let comparison_operator = match operation {
            ExtremumOperation::Max => ComparisonOp::LessThan,
            ExtremumOperation::Min => ComparisonOp::GreaterThan,
        };

        let cond1 = self.context.new_comparison(
            self.location,
            comparison_operator,
            previous_var.to_rvalue(),
            self.context.new_cast(self.location, src, previous_value.get_type()),
        );
        let compare_exchange =
            self.compare_exchange(dst, previous_var, src, order, load_ordering, false);
        let cond2 = self.cx.context.new_unary_op(
            self.location,
            UnaryOp::LogicalNegate,
            compare_exchange.get_type(),
            compare_exchange,
        );
        let cond = self.cx.context.new_binary_op(
            self.location,
            BinaryOp::LogicalAnd,
            self.cx.bool_type,
            cond1,
            cond2,
        );

        while_block.end_with_conditional(self.location, cond, while_block, after_block);

        // NOTE: since jumps were added in a place rustc does not expect, the current block in the
        // state need to be updated.
        self.switch_to_block(after_block);

        return_value.to_rvalue()
    }

    fn compare_exchange(
        &self,
        dst: RValue<'gcc>,
        cmp: LValue<'gcc>,
        src: RValue<'gcc>,
        order: AtomicOrdering,
        failure_order: AtomicOrdering,
        weak: bool,
    ) -> RValue<'gcc> {
        let size = get_maybe_pointer_size(src);
        let compare_exchange =
            self.context.get_builtin_function(format!("__atomic_compare_exchange_{}", size));
        let order = self.context.new_rvalue_from_int(self.i32_type, order.to_gcc());
        let failure_order = self.context.new_rvalue_from_int(self.i32_type, failure_order.to_gcc());
        let weak = self.context.new_rvalue_from_int(self.bool_type, weak as i32);

        let void_ptr_type = self.context.new_type::<*mut ()>();
        let volatile_void_ptr_type = void_ptr_type.make_volatile();
        let dst = self.context.new_cast(self.location, dst, volatile_void_ptr_type);
        let expected =
            self.context.new_cast(self.location, cmp.get_address(self.location), void_ptr_type);

        // NOTE: not sure why, but we have the wrong type here.
        let int_type = compare_exchange.get_param(2).to_rvalue().get_type();
        let src = self.context.new_bitcast(self.location, src, int_type);
        self.context.new_call(
            self.location,
            compare_exchange,
            &[dst, expected, src, weak, order, failure_order],
        )
    }

    pub fn assign(&self, lvalue: LValue<'gcc>, value: RValue<'gcc>) {
        self.llbb().add_assignment(self.location, lvalue, value);
    }

    fn check_call<'b>(
        &mut self,
        _typ: &str,
        func: Function<'gcc>,
        args: &'b [RValue<'gcc>],
    ) -> Cow<'b, [RValue<'gcc>]> {
        let mut all_args_match = true;
        let mut param_types = vec![];
        let param_count = func.get_param_count();
        for (index, arg) in args.iter().enumerate().take(param_count) {
            let param = func.get_param(index as i32);
            let param = param.to_rvalue().get_type();
            if param != arg.get_type() {
                all_args_match = false;
            }
            param_types.push(param);
        }

        if all_args_match {
            return Cow::Borrowed(args);
        }

        let casted_args: Vec<_> = param_types
            .into_iter()
            .zip(args.iter())
            .map(|(expected_ty, &actual_val)| {
                let actual_ty = actual_val.get_type();
                if expected_ty != actual_ty {
                    self.bitcast(actual_val, expected_ty)
                } else {
                    actual_val
                }
            })
            .collect();

        debug_assert_eq!(casted_args.len(), args.len());

        Cow::Owned(casted_args)
    }

    fn check_ptr_call<'b>(
        &mut self,
        _typ: &str,
        func_ptr: RValue<'gcc>,
        args: &'b [RValue<'gcc>],
    ) -> Cow<'b, [RValue<'gcc>]> {
        let mut all_args_match = true;
        let mut param_types = vec![];
        let gcc_func = func_ptr.get_type().dyncast_function_ptr_type().expect("function ptr");
        for (index, arg) in args.iter().enumerate().take(gcc_func.get_param_count()) {
            let param = gcc_func.get_param_type(index);
            if param != arg.get_type() {
                all_args_match = false;
            }
            param_types.push(param);
        }

        let mut on_stack_param_indices = FxHashSet::default();
        if let Some(indices) = self.on_stack_params.borrow().get(&gcc_func) {
            on_stack_param_indices.clone_from(indices);
        }

        if all_args_match {
            return Cow::Borrowed(args);
        }

        let func_name = format!("{:?}", func_ptr);

        let mut casted_args: Vec<_> = param_types
            .into_iter()
            .zip(args.iter())
            .enumerate()
            .map(|(index, (expected_ty, &actual_val))| {
                if llvm::ignore_arg_cast(&func_name, index, args.len()) {
                    return actual_val;
                }

                let actual_ty = actual_val.get_type();
                if expected_ty != actual_ty {
                    if !actual_ty.is_vector()
                        && !expected_ty.is_vector()
                        && (actual_ty.is_integral() && expected_ty.is_integral())
                        || (actual_ty.get_pointee().is_some()
                            && expected_ty.get_pointee().is_some())
                    {
                        self.context.new_cast(self.location, actual_val, expected_ty)
                    } else if on_stack_param_indices.contains(&index) {
                        let ty = actual_val.get_type();
                        // It's possible that the value behind the pointer is actually not exactly
                        // the expected type, so to go around that, we add a cast before
                        // dereferencing the value.
                        if let Some(pointee_val) = ty.get_pointee()
                            && pointee_val != expected_ty
                        {
                            let new_val = self.context.new_cast(
                                self.location,
                                actual_val,
                                expected_ty.make_pointer(),
                            );
                            new_val.dereference(self.location).to_rvalue()
                        } else {
                            actual_val.dereference(self.location).to_rvalue()
                        }
                    } else {
                        // FIXME: this condition seems wrong: it will pass when both types are not
                        // a vector.
                        assert!(
                            (!expected_ty.is_vector() || actual_ty.is_vector())
                                && (expected_ty.is_vector() || !actual_ty.is_vector()),
                            "{:?} (is vector: {}) -> {:?} (is vector: {}), Function: {:?}[{}]",
                            actual_ty,
                            actual_ty.is_vector(),
                            expected_ty,
                            expected_ty.is_vector(),
                            func_ptr,
                            index
                        );
                        // TODO(antoyo): perhaps use __builtin_convertvector for vector casting.
                        // TODO: remove bitcast now that vector types can be compared?
                        // ==> We use bitcast to avoid having to do many manual casts from e.g. __m256i to __v32qi (in
                        // the case of _mm256_aesenc_epi128).
                        self.bitcast(actual_val, expected_ty)
                    }
                } else {
                    actual_val
                }
            })
            .collect();

        // NOTE: to take into account variadic functions.
        for arg in args.iter().skip(casted_args.len()) {
            casted_args.push(*arg);
        }

        Cow::Owned(casted_args)
    }

    fn check_store(&mut self, val: RValue<'gcc>, ptr: RValue<'gcc>) -> RValue<'gcc> {
        let stored_ty = self.cx.val_ty(val);
        let stored_ptr_ty = self.cx.type_ptr_to(stored_ty);
        self.bitcast(ptr, stored_ptr_ty)
    }

    pub fn current_func(&self) -> Function<'gcc> {
        self.block.get_function()
    }

    fn function_call(
        &mut self,
        func: RValue<'gcc>,
        args: &[RValue<'gcc>],
        _funclet: Option<&Funclet>,
    ) -> RValue<'gcc> {
        // TODO(antoyo): remove when the API supports a different type for functions.
        let func: Function<'gcc> = self.cx.rvalue_as_function(func);
        let args = self.check_call("call", func, args);

        // gccjit requires to use the result of functions, even when it's not used.
        // That's why we assign the result to a local or call add_eval().
        let return_type = func.get_return_type();
        let void_type = self.context.new_type::<()>();
        let current_func = self.block.get_function();
        if return_type != void_type {
            let result = current_func.new_local(
                self.location,
                return_type,
                format!("returnValue{}", self.next_value_counter()),
            );
            self.block.add_assignment(
                self.location,
                result,
                self.cx.context.new_call(self.location, func, &args),
            );
            result.to_rvalue()
        } else {
            self.block
                .add_eval(self.location, self.cx.context.new_call(self.location, func, &args));
            // Return dummy value when not having return value.
            self.context.new_rvalue_zero(self.isize_type)
        }
    }

    fn function_ptr_call(
        &mut self,
        typ: Type<'gcc>,
        mut func_ptr: RValue<'gcc>,
        args: &[RValue<'gcc>],
        _funclet: Option<&Funclet>,
    ) -> RValue<'gcc> {
        let gcc_func = match func_ptr.get_type().dyncast_function_ptr_type() {
            Some(func) => func,
            None => {
                // NOTE: due to opaque pointers now being used, we need to cast here.
                let new_func_type = typ.dyncast_function_ptr_type().expect("function ptr");
                func_ptr = self.context.new_cast(self.location, func_ptr, typ);
                new_func_type
            }
        };
        let func_name = format!("{:?}", func_ptr);
        let previous_arg_count = args.len();
        let orig_args = args;
        let args = {
            func_ptr = llvm::adjust_function(self.context, &func_name, func_ptr, args);
            llvm::adjust_intrinsic_arguments(self, gcc_func, args.into(), &func_name)
        };
        let args_adjusted = args.len() != previous_arg_count;
        let args = self.check_ptr_call("call", func_ptr, &args);

        // gccjit requires to use the result of functions, even when it's not used.
        // That's why we assign the result to a local or call add_eval().
        let return_type = gcc_func.get_return_type();
        let void_type = self.context.new_type::<()>();
        let current_func = self.block.get_function();

        if return_type != void_type {
            let return_value = self.cx.context.new_call_through_ptr(self.location, func_ptr, &args);
            let return_value = llvm::adjust_intrinsic_return_value(
                self,
                return_value,
                &func_name,
                &args,
                args_adjusted,
                orig_args,
            );
            let result = current_func.new_local(
                self.location,
                return_value.get_type(),
                format!("ptrReturnValue{}", self.next_value_counter()),
            );
            self.block.add_assignment(self.location, result, return_value);
            result.to_rvalue()
        } else {
            #[cfg(not(feature = "master"))]
            if gcc_func.get_param_count() == 0 {
                // FIXME(antoyo): As a temporary workaround for unsupported LLVM intrinsics.
                self.block.add_eval(
                    self.location,
                    self.cx.context.new_call_through_ptr(self.location, func_ptr, &[]),
                );
            } else {
                self.block.add_eval(
                    self.location,
                    self.cx.context.new_call_through_ptr(self.location, func_ptr, &args),
                );
            }
            #[cfg(feature = "master")]
            self.block.add_eval(
                self.location,
                self.cx.context.new_call_through_ptr(self.location, func_ptr, &args),
            );
            // Return dummy value when not having return value.
            self.context.new_rvalue_zero(self.isize_type)
        }
    }

    pub fn overflow_call(
        &self,
        func: Function<'gcc>,
        args: &[RValue<'gcc>],
        _funclet: Option<&Funclet>,
    ) -> RValue<'gcc> {
        // gccjit requires to use the result of functions, even when it's not used.
        // That's why we assign the result to a local.
        let return_type = self.context.new_type::<bool>();
        let current_func = self.block.get_function();
        // TODO(antoyo): return the new_call() directly? Since the overflow function has no side-effects.
        let result = current_func.new_local(
            self.location,
            return_type,
            format!("overflowReturnValue{}", self.next_value_counter()),
        );
        self.block.add_assignment(
            self.location,
            result,
            self.cx.context.new_call(self.location, func, args),
        );
        result.to_rvalue()
    }
}

impl<'tcx> HasTyCtxt<'tcx> for Builder<'_, '_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.cx.tcx()
    }
}

impl HasDataLayout for Builder<'_, '_, '_> {
    fn data_layout(&self) -> &TargetDataLayout {
        self.cx.data_layout()
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

impl<'a, 'gcc, 'tcx> Deref for Builder<'a, 'gcc, 'tcx> {
    type Target = CodegenCx<'gcc, 'tcx>;

    fn deref<'b>(&'b self) -> &'a Self::Target {
        self.cx
    }
}

impl<'gcc, 'tcx> BackendTypes for Builder<'_, 'gcc, 'tcx> {
    type Value = <CodegenCx<'gcc, 'tcx> as BackendTypes>::Value;
    type Metadata = <CodegenCx<'gcc, 'tcx> as BackendTypes>::Metadata;
    type Function = <CodegenCx<'gcc, 'tcx> as BackendTypes>::Function;
    type BasicBlock = <CodegenCx<'gcc, 'tcx> as BackendTypes>::BasicBlock;
    type Type = <CodegenCx<'gcc, 'tcx> as BackendTypes>::Type;
    type Funclet = <CodegenCx<'gcc, 'tcx> as BackendTypes>::Funclet;

    type DIScope = <CodegenCx<'gcc, 'tcx> as BackendTypes>::DIScope;
    type DILocation = <CodegenCx<'gcc, 'tcx> as BackendTypes>::DILocation;
    type DIVariable = <CodegenCx<'gcc, 'tcx> as BackendTypes>::DIVariable;
}

fn set_rvalue_location<'a, 'gcc, 'tcx>(
    bx: &mut Builder<'a, 'gcc, 'tcx>,
    rvalue: RValue<'gcc>,
) -> RValue<'gcc> {
    if bx.location.is_some() {
        #[cfg(feature = "master")]
        rvalue.set_location(bx.location.unwrap());
    }
    rvalue
}

impl<'a, 'gcc, 'tcx> BuilderMethods<'a, 'tcx> for Builder<'a, 'gcc, 'tcx> {
    type CodegenCx = CodegenCx<'gcc, 'tcx>;

    fn build(cx: &'a CodegenCx<'gcc, 'tcx>, block: Block<'gcc>) -> Builder<'a, 'gcc, 'tcx> {
        Builder::with_cx(cx, block)
    }

    fn llbb(&self) -> Block<'gcc> {
        self.block
    }

    fn append_block(_: &'a CodegenCx<'gcc, 'tcx>, func: Function<'gcc>, name: &str) -> Block<'gcc> {
        func.new_block(name)
    }

    fn append_sibling_block(&mut self, name: &str) -> Block<'gcc> {
        let func = self.current_func();
        func.new_block(name)
    }

    fn switch_to_block(&mut self, block: Self::BasicBlock) {
        self.block = block;
    }

    fn ret_void(&mut self) {
        self.llbb().end_with_void_return(self.location)
    }

    fn ret(&mut self, mut value: RValue<'gcc>) {
        let expected_return_type = self.current_func().get_return_type();
        let value_type = value.get_type();
        if !expected_return_type.is_compatible_with(value_type) {
            // NOTE: due to opaque pointers now being used, we need to (bit)cast here.
            if self.is_native_int_type(value_type) && self.is_native_int_type(expected_return_type)
            {
                value = self.context.new_cast(self.location, value, expected_return_type);
            } else {
                value = self.context.new_bitcast(self.location, value, expected_return_type);
            }
        }
        self.llbb().end_with_return(self.location, value);
    }

    fn br(&mut self, dest: Block<'gcc>) {
        self.llbb().end_with_jump(self.location, dest)
    }

    fn cond_br(&mut self, cond: RValue<'gcc>, then_block: Block<'gcc>, else_block: Block<'gcc>) {
        self.llbb().end_with_conditional(self.location, cond, then_block, else_block)
    }

    fn switch(
        &mut self,
        value: RValue<'gcc>,
        default_block: Block<'gcc>,
        cases: impl ExactSizeIterator<Item = (u128, Block<'gcc>)>,
    ) {
        let mut gcc_cases = vec![];
        let typ = self.val_ty(value);
        // FIXME(FractalFir): This is a workaround for a libgccjit limitation.
        // Currently, libgccjit can't directly create 128 bit integers.
        // Since switch cases must be values, and casts are not constant, we can't use 128 bit switch cases.
        // In such a case, we will simply fall back to an if-ladder.
        // This *may* be slower than a native switch, but a slow working solution is better than none at all.
        if typ.is_i128(self) || typ.is_u128(self) {
            for (on_val, dest) in cases {
                let on_val = self.const_uint_big(typ, on_val);
                let is_case =
                    self.context.new_comparison(self.location, ComparisonOp::Equals, value, on_val);
                let next_block = self.current_func().new_block("case");
                self.block.end_with_conditional(self.location, is_case, dest, next_block);
                self.block = next_block;
            }
            self.block.end_with_jump(self.location, default_block);
        } else {
            for (on_val, dest) in cases {
                let on_val = self.const_uint_big(typ, on_val);
                gcc_cases.push(self.context.new_case(on_val, on_val, dest));
            }
            self.block.end_with_switch(self.location, value, default_block, &gcc_cases);
        }
    }

    #[cfg(feature = "master")]
    fn invoke(
        &mut self,
        typ: Type<'gcc>,
        fn_attrs: Option<&CodegenFnAttrs>,
        _fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        func: RValue<'gcc>,
        args: &[RValue<'gcc>],
        then: Block<'gcc>,
        catch: Block<'gcc>,
        _funclet: Option<&Funclet>,
        instance: Option<Instance<'tcx>>,
    ) -> RValue<'gcc> {
        let try_block = self.current_func().new_block("try");

        let current_block = self.block;
        self.block = try_block;
        let call = self.call(typ, fn_attrs, None, func, args, None, instance); // TODO(antoyo): use funclet here?
        self.block = current_block;

        let return_value =
            self.current_func().new_local(self.location, call.get_type(), "invokeResult");

        try_block.add_assignment(self.location, return_value, call);

        try_block.end_with_jump(self.location, then);

        if self.cleanup_blocks.borrow().contains(&catch) {
            self.block.add_try_finally(self.location, try_block, catch);
        } else {
            self.block.add_try_catch(self.location, try_block, catch);
        }

        self.block.end_with_jump(self.location, then);

        return_value.to_rvalue()
    }

    #[cfg(not(feature = "master"))]
    fn invoke(
        &mut self,
        typ: Type<'gcc>,
        fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        func: RValue<'gcc>,
        args: &[RValue<'gcc>],
        then: Block<'gcc>,
        catch: Block<'gcc>,
        _funclet: Option<&Funclet>,
        instance: Option<Instance<'tcx>>,
    ) -> RValue<'gcc> {
        let call_site = self.call(typ, fn_attrs, None, func, args, None, instance);
        let condition = self.context.new_rvalue_from_int(self.bool_type, 1);
        self.llbb().end_with_conditional(self.location, condition, then, catch);
        if let Some(_fn_abi) = fn_abi {
            // TODO(bjorn3): Apply function attributes
        }
        call_site
    }

    fn unreachable(&mut self) {
        let func = self.context.get_builtin_function("__builtin_unreachable");
        self.block.add_eval(self.location, self.context.new_call(self.location, func, &[]));
        let return_type = self.block.get_function().get_return_type();
        let void_type = self.context.new_type::<()>();
        if return_type == void_type {
            self.block.end_with_void_return(self.location)
        } else {
            let return_value =
                self.current_func().new_local(self.location, return_type, "unreachableReturn");
            self.block.end_with_return(self.location, return_value)
        }
    }

    fn add(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.gcc_add(a, b)
    }

    fn fadd(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a + b
    }

    // TODO(antoyo): should we also override the `unchecked_` versions?
    fn sub(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.gcc_sub(a, b)
    }

    fn fsub(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a - b
    }

    fn mul(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.gcc_mul(a, b)
    }

    fn fmul(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.cx.context.new_binary_op(self.location, BinaryOp::Mult, a.get_type(), a, b)
    }

    fn udiv(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.gcc_udiv(a, b)
    }

    fn exactudiv(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): poison if not exact.
        let a_type = a.get_type().to_unsigned(self);
        let a = self.gcc_int_cast(a, a_type);
        let b_type = b.get_type().to_unsigned(self);
        let b = self.gcc_int_cast(b, b_type);
        self.gcc_udiv(a, b)
    }

    fn sdiv(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.gcc_sdiv(a, b)
    }

    fn exactsdiv(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): poison if not exact.
        // FIXME(antoyo): rustc_codegen_ssa::mir::intrinsic uses different types for a and b but they
        // should be the same.
        let typ = a.get_type().to_signed(self);
        let b = self.gcc_int_cast(b, typ);
        self.gcc_sdiv(a, b)
    }

    fn fdiv(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a / b
    }

    fn urem(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.gcc_urem(a, b)
    }

    fn srem(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.gcc_srem(a, b)
    }

    fn frem(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): add check in libgccjit since using the binary operator % causes the following error:
        // during RTL pass: expand
        // libgccjit.so: error: in expmed_mode_index, at expmed.h:240
        // 0x7f0101d58dc6 expmed_mode_index
        //     ../../../gcc/gcc/expmed.h:240
        // 0x7f0101d58e35 expmed_op_cost_ptr
        //     ../../../gcc/gcc/expmed.h:262
        // 0x7f0101d594a1 sdiv_cost_ptr
        //     ../../../gcc/gcc/expmed.h:531
        // 0x7f0101d594f3 sdiv_cost
        //     ../../../gcc/gcc/expmed.h:549
        // 0x7f0101d6af7e expand_divmod(int, tree_code, machine_mode, rtx_def*, rtx_def*, rtx_def*, int, optab_methods)
        //     ../../../gcc/gcc/expmed.cc:4356
        // 0x7f0101d94f9e expand_expr_divmod
        //     ../../../gcc/gcc/expr.cc:8929
        // 0x7f0101d97a26 expand_expr_real_2(separate_ops*, rtx_def*, machine_mode, expand_modifier)
        //     ../../../gcc/gcc/expr.cc:9566
        // 0x7f0101bef6ef expand_gimple_stmt_1
        //     ../../../gcc/gcc/cfgexpand.cc:3967
        // 0x7f0101bef910 expand_gimple_stmt
        //     ../../../gcc/gcc/cfgexpand.cc:4028
        // 0x7f0101bf6ee7 expand_gimple_basic_block
        //     ../../../gcc/gcc/cfgexpand.cc:6069
        // 0x7f0101bf9194 execute
        //     ../../../gcc/gcc/cfgexpand.cc:6795
        let a_type = a.get_type();
        let a_type_unqualified = a_type.unqualified();
        if a_type.is_compatible_with(self.cx.float_type) {
            let fmodf = self.context.get_builtin_function("fmodf");
            // FIXME(antoyo): this seems to produce the wrong result.
            return self.context.new_call(self.location, fmodf, &[a, b]);
        }

        #[cfg(feature = "master")]
        match self.cx.type_kind(a_type) {
            TypeKind::Half => {
                let fmodf = self.context.get_builtin_function("fmodf");
                let f32_type = self.type_f32();
                let a = self.context.new_cast(self.location, a, f32_type);
                let b = self.context.new_cast(self.location, b, f32_type);
                let result = self.context.new_call(self.location, fmodf, &[a, b]);
                return self.context.new_cast(self.location, result, a_type);
            }
            TypeKind::Float => {
                let fmodf = self.context.get_builtin_function("fmodf");
                return self.context.new_call(self.location, fmodf, &[a, b]);
            }
            TypeKind::Double => {
                let fmod = self.context.get_builtin_function("fmod");
                return self.context.new_call(self.location, fmod, &[a, b]);
            }
            TypeKind::FP128 => {
                // TODO(antoyo): use get_simple_function_f128_2args.
                let f128_type = self.type_f128();
                let fmodf128 = self.context.new_function(
                    None,
                    gccjit::FunctionType::Extern,
                    f128_type,
                    &[
                        self.context.new_parameter(None, f128_type, "a"),
                        self.context.new_parameter(None, f128_type, "b"),
                    ],
                    "fmodf128",
                    false,
                );
                return self.context.new_call(self.location, fmodf128, &[a, b]);
            }
            _ => (),
        }

        if let Some(vector_type) = a_type_unqualified.dyncast_vector() {
            assert_eq!(a_type_unqualified, b.get_type().unqualified());

            let num_units = vector_type.get_num_units();
            let new_elements: Vec<_> = (0..num_units)
                .map(|i| {
                    let index = self.context.new_rvalue_from_long(self.cx.type_u32(), i as _);
                    let x = self.extract_element(a, index).to_rvalue();
                    let y = self.extract_element(b, index).to_rvalue();
                    self.frem(x, y)
                })
                .collect();

            return self.context.new_rvalue_from_vector(self.location, a_type, &new_elements);
        }
        assert_eq!(a_type_unqualified, self.cx.double_type);

        let fmod = self.context.get_builtin_function("fmod");
        self.context.new_call(self.location, fmod, &[a, b])
    }

    fn shl(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.gcc_shl(a, b)
    }

    fn lshr(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.gcc_lshr(a, b)
    }

    fn ashr(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): check whether behavior is an arithmetic shift for >> .
        // It seems to be if the value is signed.
        self.gcc_lshr(a, b)
    }

    fn and(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.gcc_and(a, b)
    }

    fn or(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.cx.gcc_or(a, b, self.location)
    }

    fn xor(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        set_rvalue_location(self, self.gcc_xor(a, b))
    }

    fn neg(&mut self, a: RValue<'gcc>) -> RValue<'gcc> {
        set_rvalue_location(self, self.gcc_neg(a))
    }

    fn fneg(&mut self, a: RValue<'gcc>) -> RValue<'gcc> {
        set_rvalue_location(
            self,
            self.cx.context.new_unary_op(self.location, UnaryOp::Minus, a.get_type(), a),
        )
    }

    fn not(&mut self, a: RValue<'gcc>) -> RValue<'gcc> {
        set_rvalue_location(self, self.gcc_not(a))
    }

    fn fadd_fast(&mut self, lhs: RValue<'gcc>, rhs: RValue<'gcc>) -> RValue<'gcc> {
        // NOTE: it seems like we cannot enable fast-mode for a single operation in GCC.
        set_rvalue_location(self, lhs + rhs)
    }

    fn fsub_fast(&mut self, lhs: RValue<'gcc>, rhs: RValue<'gcc>) -> RValue<'gcc> {
        // NOTE: it seems like we cannot enable fast-mode for a single operation in GCC.
        set_rvalue_location(self, lhs - rhs)
    }

    fn fmul_fast(&mut self, lhs: RValue<'gcc>, rhs: RValue<'gcc>) -> RValue<'gcc> {
        // NOTE: it seems like we cannot enable fast-mode for a single operation in GCC.
        set_rvalue_location(self, lhs * rhs)
    }

    fn fdiv_fast(&mut self, lhs: RValue<'gcc>, rhs: RValue<'gcc>) -> RValue<'gcc> {
        // NOTE: it seems like we cannot enable fast-mode for a single operation in GCC.
        set_rvalue_location(self, lhs / rhs)
    }

    fn frem_fast(&mut self, lhs: RValue<'gcc>, rhs: RValue<'gcc>) -> RValue<'gcc> {
        // NOTE: it seems like we cannot enable fast-mode for a single operation in GCC.
        let result = self.frem(lhs, rhs);
        set_rvalue_location(self, result);
        result
    }

    fn fadd_algebraic(&mut self, lhs: RValue<'gcc>, rhs: RValue<'gcc>) -> RValue<'gcc> {
        // NOTE: it seems like we cannot enable fast-mode for a single operation in GCC.
        lhs + rhs
    }

    fn fsub_algebraic(&mut self, lhs: RValue<'gcc>, rhs: RValue<'gcc>) -> RValue<'gcc> {
        // NOTE: it seems like we cannot enable fast-mode for a single operation in GCC.
        lhs - rhs
    }

    fn fmul_algebraic(&mut self, lhs: RValue<'gcc>, rhs: RValue<'gcc>) -> RValue<'gcc> {
        // NOTE: it seems like we cannot enable fast-mode for a single operation in GCC.
        lhs * rhs
    }

    fn fdiv_algebraic(&mut self, lhs: RValue<'gcc>, rhs: RValue<'gcc>) -> RValue<'gcc> {
        // NOTE: it seems like we cannot enable fast-mode for a single operation in GCC.
        lhs / rhs
    }

    fn frem_algebraic(&mut self, lhs: RValue<'gcc>, rhs: RValue<'gcc>) -> RValue<'gcc> {
        // NOTE: it seems like we cannot enable fast-mode for a single operation in GCC.
        self.frem(lhs, rhs)
    }

    fn checked_binop(
        &mut self,
        oop: OverflowOp,
        typ: Ty<'tcx>,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> (Self::Value, Self::Value) {
        self.gcc_checked_binop(oop, typ, lhs, rhs)
    }

    fn alloca(&mut self, size: Size, align: Align) -> RValue<'gcc> {
        let ty = self.cx.type_array(self.cx.type_i8(), size.bytes()).get_aligned(align.bytes());
        // TODO(antoyo): It might be better to return a LValue, but fixing the rustc API is non-trivial.
        self.current_func()
            .new_local(self.location, ty, format!("stack_var_{}", self.next_value_counter()))
            .get_address(self.location)
    }

    fn load(&mut self, pointee_ty: Type<'gcc>, ptr: RValue<'gcc>, align: Align) -> RValue<'gcc> {
        let block = self.llbb();
        let function = block.get_function();
        // NOTE(FractalFir): In some cases, we *should* skip the call to get_aligned.
        // For example, calling `get_aligned` on a i8 is pointless(since it can only be 1 aligned)
        // Calling get_aligned on a `u128`/`i128` causes the attribute to become "stacked"
        //
        // From GCCs perspective:
        // __int128_t  __attribute__((aligned(16)))  __attribute__((aligned(16)))
        // and:
        // __int128_t  __attribute__((aligned(16)))
        // are 2 distinct, incompatible types.
        //
        // So, we skip the call to `get_aligned` in such a case. *Ideally*, we could do this for all the types,
        // but the GCC APIs to facilitate this just aren't quite there yet.

        // This checks that we only skip `get_aligned` on 128 bit ints if they have the correct alignment.
        // Otherwise, this may be an under-aligned load, so we will still call get_aligned.
        let mut can_skip_align = (pointee_ty == self.cx.u128_type
            || pointee_ty == self.cx.i128_type)
            && align == self.int128_align;
        // We can skip the call to `get_aligned` for byte-sized types with alignment of 1.
        can_skip_align = can_skip_align
            || (pointee_ty == self.cx.u8_type || pointee_ty == self.cx.i8_type)
                && align.bytes() == 1;
        // Skip the call to `get_aligned` when possible.
        let aligned_type =
            if can_skip_align { pointee_ty } else { pointee_ty.get_aligned(align.bytes()) };

        let ptr = self.context.new_cast(self.location, ptr, aligned_type.make_pointer());
        // NOTE: instead of returning the dereference here, we have to assign it to a variable in
        // the current basic block. Otherwise, it could be used in another basic block, causing a
        // dereference after a drop, for instance.
        let deref = ptr.dereference(self.location).to_rvalue();
        let loaded_value = function.new_local(
            self.location,
            aligned_type,
            format!("loadedValue{}", self.next_value_counter()),
        );
        block.add_assignment(self.location, loaded_value, deref);
        loaded_value.to_rvalue()
    }

    fn volatile_load(&mut self, ty: Type<'gcc>, ptr: RValue<'gcc>) -> RValue<'gcc> {
        let ptr = self.context.new_cast(self.location, ptr, ty.make_volatile().make_pointer());
        // (FractalFir): We insert a local here, to ensure this volatile load can't move across
        // blocks.
        let local = self.current_func().new_local(self.location, ty, "volatile_tmp");
        self.block.add_assignment(self.location, local, ptr.dereference(self.location).to_rvalue());
        local.to_rvalue()
    }

    fn atomic_load(
        &mut self,
        _ty: Type<'gcc>,
        ptr: RValue<'gcc>,
        order: AtomicOrdering,
        size: Size,
    ) -> RValue<'gcc> {
        // TODO(antoyo): use ty.
        // TODO(antoyo): handle alignment.
        let atomic_load =
            self.context.get_builtin_function(format!("__atomic_load_{}", size.bytes()));
        let ordering = self.context.new_rvalue_from_int(self.i32_type, order.to_gcc());

        let volatile_const_void_ptr_type =
            self.context.new_type::<()>().make_const().make_volatile().make_pointer();
        let ptr = self.context.new_cast(self.location, ptr, volatile_const_void_ptr_type);
        self.context.new_call(self.location, atomic_load, &[ptr, ordering])
    }

    fn load_operand(
        &mut self,
        place: PlaceRef<'tcx, RValue<'gcc>>,
    ) -> OperandRef<'tcx, RValue<'gcc>> {
        assert_eq!(place.val.llextra.is_some(), place.layout.is_unsized());

        if place.layout.is_zst() {
            return OperandRef::zero_sized(place.layout);
        }

        fn scalar_load_metadata<'a, 'gcc, 'tcx>(
            bx: &mut Builder<'a, 'gcc, 'tcx>,
            load: RValue<'gcc>,
            scalar: &abi::Scalar,
        ) {
            let vr = scalar.valid_range(bx);
            match scalar.primitive() {
                abi::Primitive::Int(..) => {
                    if !scalar.is_always_valid(bx) {
                        bx.range_metadata(load, vr);
                    }
                }
                abi::Primitive::Pointer(_) if vr.start < vr.end && !vr.contains(0) => {
                    bx.nonnull_metadata(load);
                }
                _ => {}
            }
        }

        let val = if place.val.llextra.is_some() {
            // FIXME: Merge with the `else` below?
            OperandValue::Ref(place.val)
        } else if place.layout.is_gcc_immediate() {
            let load = self.load(place.layout.gcc_type(self), place.val.llval, place.val.align);
            OperandValue::Immediate(
                if let abi::BackendRepr::Scalar(ref scalar) = place.layout.backend_repr {
                    scalar_load_metadata(self, load, scalar);
                    self.to_immediate_scalar(load, *scalar)
                } else {
                    load
                },
            )
        } else if let abi::BackendRepr::ScalarPair(ref a, ref b) = place.layout.backend_repr {
            let b_offset = a.size(self).align_to(b.align(self).abi);

            let mut load = |i, scalar: &abi::Scalar, align| {
                let ptr = if i == 0 {
                    place.val.llval
                } else {
                    self.inbounds_ptradd(place.val.llval, self.const_usize(b_offset.bytes()))
                };
                let llty = place.layout.scalar_pair_element_gcc_type(self, i);
                let load = self.load(llty, ptr, align);
                scalar_load_metadata(self, load, scalar);
                if scalar.is_bool() { self.trunc(load, self.type_i1()) } else { load }
            };

            OperandValue::Pair(
                load(0, a, place.val.align),
                load(1, b, place.val.align.restrict_for_offset(b_offset)),
            )
        } else {
            OperandValue::Ref(place.val)
        };

        OperandRef { val, layout: place.layout, move_annotation: None }
    }

    fn write_operand_repeatedly(
        &mut self,
        cg_elem: OperandRef<'tcx, RValue<'gcc>>,
        count: u64,
        dest: PlaceRef<'tcx, RValue<'gcc>>,
    ) {
        let zero = self.const_usize(0);
        let count = self.const_usize(count);
        let start = dest.project_index(self, zero).val.llval;
        let end = dest.project_index(self, count).val.llval;

        let header_bb = self.append_sibling_block("repeat_loop_header");
        let body_bb = self.append_sibling_block("repeat_loop_body");
        let next_bb = self.append_sibling_block("repeat_loop_next");

        let ptr_type = start.get_type();
        let current = self.llbb().get_function().new_local(self.location, ptr_type, "loop_var");
        let current_val = current.to_rvalue();
        self.assign(current, start);

        self.br(header_bb);

        self.switch_to_block(header_bb);
        let keep_going = self.icmp(IntPredicate::IntNE, current_val, end);
        self.cond_br(keep_going, body_bb, next_bb);

        self.switch_to_block(body_bb);
        let align = dest.val.align.restrict_for_offset(dest.layout.field(self.cx(), 0).size);
        cg_elem.val.store(self, PlaceRef::new_sized_aligned(current_val, cg_elem.layout, align));

        let next = self.inbounds_gep(
            self.backend_type(cg_elem.layout),
            current.to_rvalue(),
            &[self.const_usize(1)],
        );
        self.llbb().add_assignment(self.location, current, next);
        self.br(header_bb);

        self.switch_to_block(next_bb);
    }

    fn range_metadata(&mut self, _load: RValue<'gcc>, _range: WrappingRange) {
        // TODO(antoyo)
    }

    fn nonnull_metadata(&mut self, _load: RValue<'gcc>) {
        // TODO(antoyo)
    }

    fn store(&mut self, val: RValue<'gcc>, ptr: RValue<'gcc>, align: Align) -> RValue<'gcc> {
        self.store_with_flags(val, ptr, align, MemFlags::empty())
    }

    fn store_with_flags(
        &mut self,
        val: RValue<'gcc>,
        ptr: RValue<'gcc>,
        align: Align,
        flags: MemFlags,
    ) -> RValue<'gcc> {
        let ptr = self.check_store(val, ptr);
        let destination = ptr.dereference(self.location);
        // NOTE: libgccjit does not support specifying the alignment on the assignment, so we cast
        // to type so it gets the proper alignment.
        let destination_type = destination.to_rvalue().get_type().unqualified();
        let align = if flags.contains(MemFlags::UNALIGNED) { 1 } else { align.bytes() };
        let mut modified_destination_type = destination_type.get_aligned(align);
        if flags.contains(MemFlags::VOLATILE) {
            modified_destination_type = modified_destination_type.make_volatile();
        }

        let modified_ptr =
            self.cx.context.new_cast(self.location, ptr, modified_destination_type.make_pointer());
        let modified_destination = modified_ptr.dereference(self.location);
        self.llbb().add_assignment(self.location, modified_destination, val);
        // TODO(antoyo): handle `MemFlags::NONTEMPORAL`.
        // NOTE: dummy value here since it's never used. FIXME(antoyo): API should not return a value here?
        // When adding support for NONTEMPORAL, make sure to not just emit MOVNT on x86; see the
        // LLVM backend for details.
        self.cx.context.new_rvalue_zero(self.type_i32())
    }

    fn atomic_store(
        &mut self,
        value: RValue<'gcc>,
        ptr: RValue<'gcc>,
        order: AtomicOrdering,
        size: Size,
    ) {
        // TODO(antoyo): handle alignment.
        let atomic_store =
            self.context.get_builtin_function(format!("__atomic_store_{}", size.bytes()));
        let ordering = self.context.new_rvalue_from_int(self.i32_type, order.to_gcc());
        let volatile_const_void_ptr_type =
            self.context.new_type::<()>().make_volatile().make_pointer();
        let ptr = self.context.new_cast(self.location, ptr, volatile_const_void_ptr_type);

        // FIXME(antoyo): fix libgccjit to allow comparing an integer type with an aligned integer type because
        // the following cast is required to avoid this error:
        // gcc_jit_context_new_call: mismatching types for argument 2 of function "__atomic_store_4": assignment to param arg1 (type: int) from loadedValue3577 (type: unsigned int  __attribute__((aligned(4))))
        let int_type = atomic_store.get_param(1).to_rvalue().get_type();
        let value = self.context.new_bitcast(self.location, value, int_type);
        self.llbb().add_eval(
            self.location,
            self.context.new_call(self.location, atomic_store, &[ptr, value, ordering]),
        );
    }

    fn gep(
        &mut self,
        typ: Type<'gcc>,
        ptr: RValue<'gcc>,
        indices: &[RValue<'gcc>],
    ) -> RValue<'gcc> {
        // NOTE: due to opaque pointers now being used, we need to cast here.
        let ptr = self.context.new_cast(self.location, ptr, typ.make_pointer());
        let ptr_type = ptr.get_type();
        let mut pointee_type = ptr.get_type();
        // NOTE: we cannot use array indexing here like in inbounds_gep because array indexing is
        // always considered in bounds in GCC (TODO(antoyo): to be verified).
        // So, we have to cast to a number.
        let mut result = self.context.new_bitcast(self.location, ptr, self.sizet_type);
        // FIXME(antoyo): if there were more than 1 index, this code is probably wrong and would
        // require dereferencing the pointer.
        for index in indices {
            pointee_type = pointee_type.get_pointee().expect("pointee type");
            #[cfg(feature = "master")]
            let pointee_size = {
                let size = self.cx.context.new_sizeof(pointee_type);
                self.context.new_cast(self.location, size, index.get_type())
            };
            #[cfg(not(feature = "master"))]
            let pointee_size =
                self.context.new_rvalue_from_int(index.get_type(), pointee_type.get_size() as i32);
            result = result + self.gcc_int_cast(*index * pointee_size, self.sizet_type);
        }
        self.context.new_bitcast(self.location, result, ptr_type)
    }

    fn inbounds_gep(
        &mut self,
        typ: Type<'gcc>,
        ptr: RValue<'gcc>,
        indices: &[RValue<'gcc>],
    ) -> RValue<'gcc> {
        // NOTE: due to opaque pointers now being used, we need to cast here.
        let ptr = self.context.new_cast(self.location, ptr, typ.make_pointer());
        // NOTE: array indexing is always considered in bounds in GCC (TODO(antoyo): to be verified).
        let mut indices = indices.iter();
        let index = indices.next().expect("first index in inbounds_gep");
        let mut result = self.context.new_array_access(self.location, ptr, *index);
        for index in indices {
            result = self.context.new_array_access(self.location, result, *index);
        }
        result.get_address(self.location)
    }

    /* Casts */
    fn trunc(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): check that it indeed truncate the value.
        self.gcc_int_cast(value, dest_ty)
    }

    fn sext(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): check that it indeed sign extend the value.
        if dest_ty.dyncast_vector().is_some() {
            // TODO(antoyo): nothing to do as it is only for LLVM?
            return value;
        }
        self.context.new_cast(self.location, value, dest_ty)
    }

    fn fptoui(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        set_rvalue_location(self, self.gcc_float_to_uint_cast(value, dest_ty))
    }

    fn fptosi(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        set_rvalue_location(self, self.gcc_float_to_int_cast(value, dest_ty))
    }

    fn uitofp(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        set_rvalue_location(self, self.gcc_uint_to_float_cast(value, dest_ty))
    }

    fn sitofp(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        set_rvalue_location(self, self.gcc_int_to_float_cast(value, dest_ty))
    }

    fn fptrunc(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): make sure it truncates.
        set_rvalue_location(self, self.context.new_cast(self.location, value, dest_ty))
    }

    fn fpext(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        set_rvalue_location(self, self.context.new_cast(self.location, value, dest_ty))
    }

    fn ptrtoint(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        let usize_value = self.cx.context.new_cast(None, value, self.cx.type_isize());
        self.intcast(usize_value, dest_ty, false)
    }

    fn inttoptr(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        let usize_value = self.intcast(value, self.cx.type_isize(), false);
        self.cx.context.new_cast(None, usize_value, dest_ty)
    }

    fn bitcast(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        self.cx.const_bitcast(value, dest_ty)
    }

    fn intcast(
        &mut self,
        mut value: RValue<'gcc>,
        dest_typ: Type<'gcc>,
        is_signed: bool,
    ) -> RValue<'gcc> {
        let value_type = value.get_type();
        if is_signed && !value_type.is_signed(self.cx) {
            let signed_type = value_type.to_signed(self.cx);
            value = self.gcc_int_cast(value, signed_type);
        } else if !is_signed && value_type.is_signed(self.cx) {
            let unsigned_type = value_type.to_unsigned(self.cx);
            value = self.gcc_int_cast(value, unsigned_type);
        }

        self.gcc_int_cast(value, dest_typ)
    }

    fn pointercast(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        let val_type = value.get_type();
        match (type_is_pointer(val_type), type_is_pointer(dest_ty)) {
            (false, true) => {
                // NOTE: Projecting a field of a pointer type will attempt a cast from a signed char to
                // a pointer, which is not supported by gccjit.
                self.cx.context.new_cast(
                    self.location,
                    self.inttoptr(value, val_type.make_pointer()),
                    dest_ty,
                )
            }
            (false, false) => {
                // When they are not pointers, we want a transmute (or reinterpret_cast).
                self.bitcast(value, dest_ty)
            }
            (true, true) => self.cx.context.new_cast(self.location, value, dest_ty),
            (true, false) => unimplemented!(),
        }
    }

    /* Comparisons */
    fn icmp(&mut self, op: IntPredicate, lhs: RValue<'gcc>, rhs: RValue<'gcc>) -> RValue<'gcc> {
        self.gcc_icmp(op, lhs, rhs)
    }

    fn fcmp(&mut self, op: RealPredicate, lhs: RValue<'gcc>, rhs: RValue<'gcc>) -> RValue<'gcc> {
        // LLVM has a concept of "unordered compares", where eg ULT returns true if either the two
        // arguments are unordered (i.e. either is NaN), or the lhs is less than the rhs. GCC does
        // not natively have this concept, so in some cases we must manually handle NaNs
        let must_handle_nan = match op {
            RealPredicate::RealPredicateFalse => unreachable!(),
            RealPredicate::RealOEQ => false,
            RealPredicate::RealOGT => false,
            RealPredicate::RealOGE => false,
            RealPredicate::RealOLT => false,
            RealPredicate::RealOLE => false,
            RealPredicate::RealONE => false,
            RealPredicate::RealORD => unreachable!(),
            RealPredicate::RealUNO => unreachable!(),
            RealPredicate::RealUEQ => false,
            RealPredicate::RealUGT => true,
            RealPredicate::RealUGE => true,
            RealPredicate::RealULT => true,
            RealPredicate::RealULE => true,
            RealPredicate::RealUNE => false,
            RealPredicate::RealPredicateTrue => unreachable!(),
        };

        let cmp = self.context.new_comparison(self.location, op.to_gcc_comparison(), lhs, rhs);

        if must_handle_nan {
            let is_nan = self.context.new_binary_op(
                self.location,
                BinaryOp::LogicalOr,
                self.cx.bool_type,
                // compare a value to itself to check whether it is NaN
                self.context.new_comparison(self.location, ComparisonOp::NotEquals, lhs, lhs),
                self.context.new_comparison(self.location, ComparisonOp::NotEquals, rhs, rhs),
            );

            self.context.new_binary_op(
                self.location,
                BinaryOp::LogicalOr,
                self.cx.bool_type,
                is_nan,
                cmp,
            )
        } else {
            cmp
        }
    }

    /* Miscellaneous instructions */
    fn memcpy(
        &mut self,
        dst: RValue<'gcc>,
        _dst_align: Align,
        src: RValue<'gcc>,
        _src_align: Align,
        size: RValue<'gcc>,
        flags: MemFlags,
        _tt: Option<rustc_ast::expand::typetree::FncTree>, // Autodiff TypeTrees are LLVM-only, ignored in GCC backend
    ) {
        assert!(!flags.contains(MemFlags::NONTEMPORAL), "non-temporal memcpy not supported");
        let size = self.intcast(size, self.type_size_t(), false);
        let _is_volatile = flags.contains(MemFlags::VOLATILE);
        let dst = self.pointercast(dst, self.type_i8p());
        let src = self.pointercast(src, self.type_ptr_to(self.type_void()));
        let memcpy = self.context.get_builtin_function("memcpy");
        // TODO(antoyo): handle aligns and is_volatile.
        self.block.add_eval(
            self.location,
            self.context.new_call(self.location, memcpy, &[dst, src, size]),
        );
    }

    fn memmove(
        &mut self,
        dst: RValue<'gcc>,
        _dst_align: Align,
        src: RValue<'gcc>,
        _src_align: Align,
        size: RValue<'gcc>,
        flags: MemFlags,
    ) {
        assert!(!flags.contains(MemFlags::NONTEMPORAL), "non-temporal memmove not supported");
        let size = self.intcast(size, self.type_size_t(), false);
        let _is_volatile = flags.contains(MemFlags::VOLATILE);
        let dst = self.pointercast(dst, self.type_i8p());
        let src = self.pointercast(src, self.type_ptr_to(self.type_void()));

        let memmove = self.context.get_builtin_function("memmove");
        // TODO(antoyo): handle is_volatile.
        self.block.add_eval(
            self.location,
            self.context.new_call(self.location, memmove, &[dst, src, size]),
        );
    }

    fn memset(
        &mut self,
        ptr: RValue<'gcc>,
        fill_byte: RValue<'gcc>,
        size: RValue<'gcc>,
        _align: Align,
        flags: MemFlags,
    ) {
        assert!(!flags.contains(MemFlags::NONTEMPORAL), "non-temporal memset not supported");
        let _is_volatile = flags.contains(MemFlags::VOLATILE);
        let ptr = self.pointercast(ptr, self.type_i8p());
        let memset = self.context.get_builtin_function("memset");
        // TODO(antoyo): handle align and is_volatile.
        let fill_byte = self.context.new_cast(self.location, fill_byte, self.i32_type);
        let size = self.intcast(size, self.type_size_t(), false);
        self.block.add_eval(
            self.location,
            self.context.new_call(self.location, memset, &[ptr, fill_byte, size]),
        );
    }

    fn select(
        &mut self,
        cond: RValue<'gcc>,
        then_val: RValue<'gcc>,
        mut else_val: RValue<'gcc>,
    ) -> RValue<'gcc> {
        let func = self.current_func();
        let variable = func.new_local(self.location, then_val.get_type(), "selectVar");
        let then_block = func.new_block("then");
        let else_block = func.new_block("else");
        let after_block = func.new_block("after");
        self.llbb().end_with_conditional(self.location, cond, then_block, else_block);

        then_block.add_assignment(self.location, variable, then_val);
        then_block.end_with_jump(self.location, after_block);

        if !then_val.get_type().is_compatible_with(else_val.get_type()) {
            else_val = self.context.new_cast(self.location, else_val, then_val.get_type());
        }
        else_block.add_assignment(self.location, variable, else_val);
        else_block.end_with_jump(self.location, after_block);

        // NOTE: since jumps were added in a place rustc does not expect, the current block in the
        // state need to be updated.
        self.switch_to_block(after_block);

        variable.to_rvalue()
    }

    #[allow(dead_code)]
    fn va_arg(&mut self, _list: RValue<'gcc>, _ty: Type<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    #[cfg(feature = "master")]
    fn extract_element(&mut self, vec: RValue<'gcc>, idx: RValue<'gcc>) -> RValue<'gcc> {
        self.context.new_vector_access(self.location, vec, idx).to_rvalue()
    }

    #[cfg(not(feature = "master"))]
    fn extract_element(&mut self, vec: RValue<'gcc>, idx: RValue<'gcc>) -> RValue<'gcc> {
        let vector_type = vec
            .get_type()
            .unqualified()
            .dyncast_vector()
            .expect("Called extract_element on a non-vector type");
        let element_type = vector_type.get_element_type();
        let vec_num_units = vector_type.get_num_units();
        let array_type =
            self.context.new_array_type(self.location, element_type, vec_num_units as u64);
        let array = self.context.new_bitcast(self.location, vec, array_type).to_rvalue();
        self.context.new_array_access(self.location, array, idx).to_rvalue()
    }

    fn vector_splat(&mut self, _num_elts: usize, _elt: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    fn extract_value(&mut self, aggregate_value: RValue<'gcc>, idx: u64) -> RValue<'gcc> {
        // FIXME(antoyo): it would be better if the API only called this on struct, not on arrays.
        assert_eq!(idx as usize as u64, idx);
        let value_type = aggregate_value.get_type();

        if value_type.dyncast_array().is_some() {
            let index = self
                .context
                .new_rvalue_from_long(self.u64_type, i64::try_from(idx).expect("i64::try_from"));
            let element = self.context.new_array_access(self.location, aggregate_value, index);
            element.get_address(self.location)
        } else if value_type.dyncast_vector().is_some() {
            panic!();
        } else if let Some(struct_type) = value_type.is_struct() {
            aggregate_value
                .access_field(self.location, struct_type.get_field(idx as i32))
                .to_rvalue()
        } else {
            panic!("Unexpected type {:?}", value_type);
        }
    }

    fn insert_value(
        &mut self,
        aggregate_value: RValue<'gcc>,
        value: RValue<'gcc>,
        idx: u64,
    ) -> RValue<'gcc> {
        // FIXME(antoyo): it would be better if the API only called this on struct, not on arrays.
        assert_eq!(idx as usize as u64, idx);
        let value_type = aggregate_value.get_type();

        let new_val = self.current_func().new_local(None, value_type, "aggregate_value");
        self.block.add_assignment(None, new_val, aggregate_value);

        let lvalue = if value_type.dyncast_array().is_some() {
            let index = self
                .context
                .new_rvalue_from_long(self.u64_type, i64::try_from(idx).expect("i64::try_from"));
            self.context.new_array_access(self.location, new_val, index)
        } else if value_type.dyncast_vector().is_some() {
            panic!();
        } else if let Some(struct_type) = value_type.is_struct() {
            new_val.access_field(None, struct_type.get_field(idx as i32))
        } else {
            panic!("Unexpected type {:?}", value_type);
        };

        let lvalue_type = lvalue.to_rvalue().get_type();
        let value =
            // NOTE: sometimes, rustc will create a value with the wrong type.
            if lvalue_type != value.get_type() {
                self.context.new_cast(self.location, value, lvalue_type)
            }
            else {
                value
            };

        self.llbb().add_assignment(self.location, lvalue, value);

        new_val.to_rvalue()
    }

    fn set_personality_fn(&mut self, _personality: Function<'gcc>) {
        #[cfg(feature = "master")]
        self.current_func().set_personality_function(_personality);
    }

    #[cfg(feature = "master")]
    fn cleanup_landing_pad(&mut self, pers_fn: Function<'gcc>) -> (RValue<'gcc>, RValue<'gcc>) {
        self.set_personality_fn(pers_fn);

        // NOTE: insert the current block in a variable so that a later call to invoke knows to
        // generate a try/finally instead of a try/catch for this block.
        self.cleanup_blocks.borrow_mut().insert(self.block);

        let eh_pointer_builtin =
            self.cx.context.get_target_builtin_function("__builtin_eh_pointer");
        let zero = self.cx.context.new_rvalue_zero(self.int_type);
        let ptr = self.cx.context.new_call(self.location, eh_pointer_builtin, &[zero]);

        let value1_type = self.u8_type.make_pointer();
        let ptr = self.cx.context.new_cast(self.location, ptr, value1_type);
        let value1 = ptr;
        let value2 = zero; // TODO(antoyo): set the proper value here (the type of exception?).

        (value1, value2)
    }

    #[cfg(not(feature = "master"))]
    fn cleanup_landing_pad(&mut self, _pers_fn: Function<'gcc>) -> (RValue<'gcc>, RValue<'gcc>) {
        let value1 = self
            .current_func()
            .new_local(self.location, self.u8_type.make_pointer(), "landing_pad0")
            .to_rvalue();
        let value2 =
            self.current_func().new_local(self.location, self.i32_type, "landing_pad1").to_rvalue();
        (value1, value2)
    }

    fn filter_landing_pad(&mut self, pers_fn: Function<'gcc>) {
        // TODO(antoyo): generate the correct landing pad
        self.cleanup_landing_pad(pers_fn);
    }

    #[cfg(feature = "master")]
    fn resume(&mut self, exn0: RValue<'gcc>, _exn1: RValue<'gcc>) {
        let exn_type = exn0.get_type();
        let exn = self.context.new_cast(self.location, exn0, exn_type);
        let unwind_resume = self.context.get_target_builtin_function("__builtin_unwind_resume");
        self.llbb()
            .add_eval(self.location, self.context.new_call(self.location, unwind_resume, &[exn]));
        self.unreachable();
    }

    #[cfg(not(feature = "master"))]
    fn resume(&mut self, _exn0: RValue<'gcc>, _exn1: RValue<'gcc>) {
        self.unreachable();
    }

    fn cleanup_pad(&mut self, _parent: Option<RValue<'gcc>>, _args: &[RValue<'gcc>]) -> Funclet {
        unimplemented!();
    }

    fn cleanup_ret(&mut self, _funclet: &Funclet, _unwind: Option<Block<'gcc>>) {
        unimplemented!();
    }

    fn catch_pad(&mut self, _parent: RValue<'gcc>, _args: &[RValue<'gcc>]) -> Funclet {
        unimplemented!();
    }

    fn catch_switch(
        &mut self,
        _parent: Option<RValue<'gcc>>,
        _unwind: Option<Block<'gcc>>,
        _handlers: &[Block<'gcc>],
    ) -> RValue<'gcc> {
        unimplemented!();
    }

    // Atomic Operations
    fn atomic_cmpxchg(
        &mut self,
        dst: RValue<'gcc>,
        cmp: RValue<'gcc>,
        src: RValue<'gcc>,
        order: AtomicOrdering,
        failure_order: AtomicOrdering,
        weak: bool,
    ) -> (RValue<'gcc>, RValue<'gcc>) {
        let expected = self.current_func().new_local(None, cmp.get_type(), "expected");
        self.llbb().add_assignment(None, expected, cmp);
        // NOTE: gcc doesn't support a failure memory model that is stronger than the success
        // memory model.
        let order = if failure_order as i32 > order as i32 { failure_order } else { order };
        let success = self.compare_exchange(dst, expected, src, order, failure_order, weak);

        // NOTE: since success contains the call to the intrinsic, it must be added to the basic block before
        // expected so that we store expected after the call.
        let success_var = self.current_func().new_local(self.location, self.bool_type, "success");
        self.llbb().add_assignment(self.location, success_var, success);

        (expected.to_rvalue(), success_var.to_rvalue())
    }

    fn atomic_rmw(
        &mut self,
        op: AtomicRmwBinOp,
        dst: RValue<'gcc>,
        src: RValue<'gcc>,
        order: AtomicOrdering,
        ret_ptr: bool,
    ) -> RValue<'gcc> {
        let size = get_maybe_pointer_size(src);
        let name = match op {
            AtomicRmwBinOp::AtomicXchg => format!("__atomic_exchange_{}", size),
            AtomicRmwBinOp::AtomicAdd => format!("__atomic_fetch_add_{}", size),
            AtomicRmwBinOp::AtomicSub => format!("__atomic_fetch_sub_{}", size),
            AtomicRmwBinOp::AtomicAnd => format!("__atomic_fetch_and_{}", size),
            AtomicRmwBinOp::AtomicNand => format!("__atomic_fetch_nand_{}", size),
            AtomicRmwBinOp::AtomicOr => format!("__atomic_fetch_or_{}", size),
            AtomicRmwBinOp::AtomicXor => format!("__atomic_fetch_xor_{}", size),
            AtomicRmwBinOp::AtomicMax => {
                return self.atomic_extremum(ExtremumOperation::Max, dst, src, order);
            }
            AtomicRmwBinOp::AtomicMin => {
                return self.atomic_extremum(ExtremumOperation::Min, dst, src, order);
            }
            AtomicRmwBinOp::AtomicUMax => {
                return self.atomic_extremum(ExtremumOperation::Max, dst, src, order);
            }
            AtomicRmwBinOp::AtomicUMin => {
                return self.atomic_extremum(ExtremumOperation::Min, dst, src, order);
            }
        };

        let atomic_function = self.context.get_builtin_function(name);
        let order = self.context.new_rvalue_from_int(self.i32_type, order.to_gcc());

        // FIXME: If `ret_ptr` is true and `src` is an integer, we should really tell GCC
        // that this is a pointer operation that needs to preserve provenance -- but like LLVM,
        // GCC does not currently seems to support that.
        let void_ptr_type = self.context.new_type::<*mut ()>();
        let volatile_void_ptr_type = void_ptr_type.make_volatile();
        let dst = self.context.new_cast(self.location, dst, volatile_void_ptr_type);
        // FIXME(antoyo): not sure why, but we have the wrong type here.
        let new_src_type = atomic_function.get_param(1).to_rvalue().get_type();
        let src = self.context.new_bitcast(self.location, src, new_src_type);
        let res = self.context.new_call(self.location, atomic_function, &[dst, src, order]);
        let res_type = if ret_ptr { void_ptr_type } else { src.get_type() };
        self.context.new_cast(self.location, res, res_type)
    }

    fn atomic_fence(&mut self, order: AtomicOrdering, scope: SynchronizationScope) {
        let name = match scope {
            SynchronizationScope::SingleThread => "__atomic_signal_fence",
            SynchronizationScope::CrossThread => "__atomic_thread_fence",
        };
        let thread_fence = self.context.get_builtin_function(name);
        let order = self.context.new_rvalue_from_int(self.i32_type, order.to_gcc());
        self.llbb()
            .add_eval(self.location, self.context.new_call(self.location, thread_fence, &[order]));
    }

    fn set_invariant_load(&mut self, load: RValue<'gcc>) {
        // NOTE: Hack to consider vtable function pointer as non-global-variable function pointer.
        self.normal_function_addresses.borrow_mut().insert(load);
        // TODO(antoyo)
    }

    fn lifetime_start(&mut self, _ptr: RValue<'gcc>, _size: Size) {
        // TODO(antoyo)
    }

    fn lifetime_end(&mut self, _ptr: RValue<'gcc>, _size: Size) {
        // TODO(antoyo)
    }

    fn call(
        &mut self,
        typ: Type<'gcc>,
        _fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        func: RValue<'gcc>,
        args: &[RValue<'gcc>],
        funclet: Option<&Funclet>,
        _instance: Option<Instance<'tcx>>,
    ) -> RValue<'gcc> {
        // FIXME(antoyo): remove when having a proper API.
        let gcc_func = unsafe { std::mem::transmute::<RValue<'gcc>, Function<'gcc>>(func) };
        let call = if self.functions.borrow().values().any(|value| *value == gcc_func) {
            self.function_call(func, args, funclet)
        } else {
            // If it's a not function that was defined, it's a function pointer.
            self.function_ptr_call(typ, func, args, funclet)
        };
        if let Some(_fn_abi) = fn_abi {
            // TODO(bjorn3): Apply function attributes
        }
        call
    }

    fn tail_call(
        &mut self,
        _llty: Self::Type,
        _fn_attrs: Option<&CodegenFnAttrs>,
        _fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        _llfn: Self::Value,
        _args: &[Self::Value],
        _funclet: Option<&Self::Funclet>,
        _instance: Option<Instance<'tcx>>,
    ) {
        // FIXME: implement support for explicit tail calls like rustc_codegen_llvm.
        self.tcx.dcx().emit_fatal(errors::ExplicitTailCallsUnsupported);
    }

    fn zext(&mut self, value: RValue<'gcc>, dest_typ: Type<'gcc>) -> RValue<'gcc> {
        // FIXME(antoyo): this does not zero-extend.
        self.gcc_int_cast(value, dest_typ)
    }

    fn cx(&self) -> &CodegenCx<'gcc, 'tcx> {
        self.cx
    }

    fn apply_attrs_to_cleanup_callsite(&mut self, _llret: RValue<'gcc>) {
        // FIXME(bjorn3): implement
    }

    fn set_span(&mut self, _span: Span) {}

    fn from_immediate(&mut self, val: Self::Value) -> Self::Value {
        if self.cx().val_ty(val) == self.cx().type_i1() {
            self.zext(val, self.cx().type_i8())
        } else {
            val
        }
    }

    fn to_immediate_scalar(&mut self, val: Self::Value, scalar: abi::Scalar) -> Self::Value {
        if scalar.is_bool() {
            return self.unchecked_utrunc(val, self.cx().type_i1());
        }
        val
    }

    fn fptoui_sat(&mut self, val: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        self.fptoint_sat(false, val, dest_ty)
    }

    fn fptosi_sat(&mut self, val: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        self.fptoint_sat(true, val, dest_ty)
    }
}

impl<'a, 'gcc, 'tcx> Builder<'a, 'gcc, 'tcx> {
    fn fptoint_sat(
        &mut self,
        signed: bool,
        val: RValue<'gcc>,
        dest_ty: Type<'gcc>,
    ) -> RValue<'gcc> {
        let src_ty = self.cx.val_ty(val);
        let (float_ty, int_ty) = if self.cx.type_kind(src_ty) == TypeKind::Vector {
            assert_eq!(self.cx.vector_length(src_ty), self.cx.vector_length(dest_ty));
            (self.cx.element_type(src_ty), self.cx.element_type(dest_ty))
        } else {
            (src_ty, dest_ty)
        };

        // FIXME(jistone): the following was originally the fallback SSA implementation, before LLVM 13
        // added native `fptosi.sat` and `fptoui.sat` conversions, but it was used by GCC as well.
        // Now that LLVM always relies on its own, the code has been moved to GCC, but the comments are
        // still LLVM-specific. This should be updated, and use better GCC specifics if possible.

        let int_width = self.cx.int_width(int_ty);
        let float_width = self.cx.float_width(float_ty);
        // LLVM's fpto[su]i returns undef when the input val is infinite, NaN, or does not fit into the
        // destination integer type after rounding towards zero. This `undef` value can cause UB in
        // safe code (see issue #10184), so we implement a saturating conversion on top of it:
        // Semantically, the mathematical value of the input is rounded towards zero to the next
        // mathematical integer, and then the result is clamped into the range of the destination
        // integer type. Positive and negative infinity are mapped to the maximum and minimum value of
        // the destination integer type. NaN is mapped to 0.
        //
        // Define f_min and f_max as the largest and smallest (finite) floats that are exactly equal to
        // a value representable in int_ty.
        // They are exactly equal to int_ty::{MIN,MAX} if float_ty has enough significand bits.
        // Otherwise, int_ty::MAX must be rounded towards zero, as it is one less than a power of two.
        // int_ty::MIN, however, is either zero or a negative power of two and is thus exactly
        // representable. Note that this only works if float_ty's exponent range is sufficiently large.
        // f16 or 256 bit integers would break this property. Right now the smallest float type is f32
        // with exponents ranging up to 127, which is barely enough for i128::MIN = -2^127.
        // On the other hand, f_max works even if int_ty::MAX is greater than float_ty::MAX. Because
        // we're rounding towards zero, we just get float_ty::MAX (which is always an integer).
        // This already happens today with u128::MAX = 2^128 - 1 > f32::MAX.
        let int_max = |signed: bool, int_width: u64| -> u128 {
            let shift_amount = 128 - int_width;
            if signed { i128::MAX as u128 >> shift_amount } else { u128::MAX >> shift_amount }
        };
        let int_min = |signed: bool, int_width: u64| -> i128 {
            if signed { i128::MIN >> (128 - int_width) } else { 0 }
        };

        let compute_clamp_bounds_single = |signed: bool, int_width: u64| -> (u128, u128) {
            let rounded_min =
                ieee::Single::from_i128_r(int_min(signed, int_width), Round::TowardZero);
            assert_eq!(rounded_min.status, Status::OK);
            let rounded_max =
                ieee::Single::from_u128_r(int_max(signed, int_width), Round::TowardZero);
            assert!(rounded_max.value.is_finite());
            (rounded_min.value.to_bits(), rounded_max.value.to_bits())
        };
        let compute_clamp_bounds_double = |signed: bool, int_width: u64| -> (u128, u128) {
            let rounded_min =
                ieee::Double::from_i128_r(int_min(signed, int_width), Round::TowardZero);
            assert_eq!(rounded_min.status, Status::OK);
            let rounded_max =
                ieee::Double::from_u128_r(int_max(signed, int_width), Round::TowardZero);
            assert!(rounded_max.value.is_finite());
            (rounded_min.value.to_bits(), rounded_max.value.to_bits())
        };
        // To implement saturation, we perform the following steps:
        //
        // 1. Cast val to an integer with fpto[su]i. This may result in undef.
        // 2. Compare val to f_min and f_max, and use the comparison results to select:
        //  a) int_ty::MIN if val < f_min or val is NaN
        //  b) int_ty::MAX if val > f_max
        //  c) the result of fpto[su]i otherwise
        // 3. If val is NaN, return 0.0, otherwise return the result of step 2.
        //
        // This avoids resulting undef because values in range [f_min, f_max] by definition fit into the
        // destination type. It creates an undef temporary, but *producing* undef is not UB. Our use of
        // undef does not introduce any non-determinism either.
        // More importantly, the above procedure correctly implements saturating conversion.
        // Proof (sketch):
        // If val is NaN, 0 is returned by definition.
        // Otherwise, val is finite or infinite and thus can be compared with f_min and f_max.
        // This yields three cases to consider:
        // (1) if val in [f_min, f_max], the result of fpto[su]i is returned, which agrees with
        //     saturating conversion for inputs in that range.
        // (2) if val > f_max, then val is larger than int_ty::MAX. This holds even if f_max is rounded
        //     (i.e., if f_max < int_ty::MAX) because in those cases, nextUp(f_max) is already larger
        //     than int_ty::MAX. Because val is larger than int_ty::MAX, the return value of int_ty::MAX
        //     is correct.
        // (3) if val < f_min, then val is smaller than int_ty::MIN. As shown earlier, f_min exactly equals
        //     int_ty::MIN and therefore the return value of int_ty::MIN is correct.
        // QED.

        let float_bits_to_llval = |bx: &mut Self, bits| {
            let bits_llval = match float_width {
                32 => bx.cx().const_u32(bits as u32),
                64 => bx.cx().const_u64(bits as u64),
                n => bug!("unsupported float width {}", n),
            };
            bx.bitcast(bits_llval, float_ty)
        };
        let (f_min, f_max) = match float_width {
            32 => compute_clamp_bounds_single(signed, int_width),
            64 => compute_clamp_bounds_double(signed, int_width),
            n => bug!("unsupported float width {}", n),
        };
        let f_min = float_bits_to_llval(self, f_min);
        let f_max = float_bits_to_llval(self, f_max);
        let int_max = self.cx.const_uint_big(int_ty, int_max(signed, int_width));
        let int_min = self.cx.const_uint_big(int_ty, int_min(signed, int_width) as u128);
        let zero = self.cx.const_uint(int_ty, 0);

        // If we're working with vectors, constants must be "splatted": the constant is duplicated
        // into each lane of the vector.  The algorithm stays the same, we are just using the
        // same constant across all lanes.
        let maybe_splat = |bx: &mut Self, val| {
            if bx.cx().type_kind(dest_ty) == TypeKind::Vector {
                bx.vector_splat(bx.vector_length(dest_ty), val)
            } else {
                val
            }
        };
        let f_min = maybe_splat(self, f_min);
        let f_max = maybe_splat(self, f_max);
        let int_max = maybe_splat(self, int_max);
        let int_min = maybe_splat(self, int_min);
        let zero = maybe_splat(self, zero);

        // Step 1 ...
        let fptosui_result =
            if signed { self.fptosi(val, dest_ty) } else { self.fptoui(val, dest_ty) };
        let less_or_nan = self.fcmp(RealPredicate::RealULT, val, f_min);
        let greater = self.fcmp(RealPredicate::RealOGT, val, f_max);

        // Step 2: We use two comparisons and two selects, with %s1 being the
        // result:
        //     %less_or_nan = fcmp ult %val, %f_min
        //     %greater = fcmp olt %val, %f_max
        //     %s0 = select %less_or_nan, int_ty::MIN, %fptosi_result
        //     %s1 = select %greater, int_ty::MAX, %s0
        // Note that %less_or_nan uses an *unordered* comparison. This
        // comparison is true if the operands are not comparable (i.e., if val is
        // NaN). The unordered comparison ensures that s1 becomes int_ty::MIN if
        // val is NaN.
        //
        // Performance note: Unordered comparison can be lowered to a "flipped"
        // comparison and a negation, and the negation can be merged into the
        // select. Therefore, it not necessarily any more expensive than an
        // ordered ("normal") comparison. Whether these optimizations will be
        // performed is ultimately up to the backend, but at least x86 does
        // perform them.
        let s0 = self.select(less_or_nan, int_min, fptosui_result);
        let s1 = self.select(greater, int_max, s0);

        // Step 3: NaN replacement.
        // For unsigned types, the above step already yielded int_ty::MIN == 0 if val is NaN.
        // Therefore we only need to execute this step for signed integer types.
        if signed {
            // LLVM has no isNaN predicate, so we use (val == val) instead
            let cmp = self.fcmp(RealPredicate::RealOEQ, val, val);
            self.select(cmp, s1, zero)
        } else {
            s1
        }
    }

    #[cfg(feature = "master")]
    pub fn shuffle_vector(
        &mut self,
        v1: RValue<'gcc>,
        v2: RValue<'gcc>,
        mask: RValue<'gcc>,
    ) -> RValue<'gcc> {
        // NOTE: if the `mask` is a constant value, the following code will copy it in many places,
        // which will make GCC create a lot (+4000) local variables in some cases.
        // So we assign it to an explicit local variable once to avoid this.
        let func = self.current_func();
        let mask_var = func.new_local(self.location, mask.get_type(), "mask");
        let block = self.block;
        block.add_assignment(self.location, mask_var, mask);
        let mask = mask_var.to_rvalue();

        // TODO(antoyo): use a recursive unqualified() here.
        let vector_type = v1.get_type().unqualified().dyncast_vector().expect("vector type");
        let element_type = vector_type.get_element_type();
        let vec_num_units = vector_type.get_num_units();

        let mask_element_type = if element_type.is_integral() {
            element_type
        } else {
            #[cfg(feature = "master")]
            {
                self.cx.type_ix(element_type.get_size() as u64 * 8)
            }
            #[cfg(not(feature = "master"))]
            self.int_type
        };

        // NOTE: this condition is needed because we call shuffle_vector in the implementation of
        // simd_gather.
        let mut mask_elements = if let Some(vector_type) = mask.get_type().dyncast_vector() {
            let mask_num_units = vector_type.get_num_units();
            let mut mask_elements = vec![];
            for i in 0..mask_num_units {
                let index = self.context.new_rvalue_from_long(self.cx.type_u32(), i as _);
                mask_elements.push(self.context.new_cast(
                    self.location,
                    self.extract_element(mask, index).to_rvalue(),
                    mask_element_type,
                ));
            }
            mask_elements
        } else {
            let struct_type = mask.get_type().is_struct().expect("mask should be of struct type");
            let mask_num_units = struct_type.get_field_count();
            let mut mask_elements = vec![];
            for i in 0..mask_num_units {
                let field = struct_type.get_field(i as i32);
                mask_elements.push(self.context.new_cast(
                    self.location,
                    mask.access_field(self.location, field).to_rvalue(),
                    mask_element_type,
                ));
            }
            mask_elements
        };
        let mask_num_units = mask_elements.len();

        // NOTE: the mask needs to be the same length as the input vectors, so add the missing
        // elements in the mask if needed.
        for _ in mask_num_units..vec_num_units {
            mask_elements.push(self.context.new_rvalue_zero(mask_element_type));
        }

        let result_type = self.context.new_vector_type(element_type, mask_num_units as u64);
        let (v1, v2) = if vec_num_units < mask_num_units {
            // NOTE: the mask needs to be the same length as the input vectors, so join the 2
            // vectors and create a dummy second vector.
            let mut elements = vec![];
            for i in 0..vec_num_units {
                elements.push(
                    self.context
                        .new_vector_access(
                            self.location,
                            v1,
                            self.context.new_rvalue_from_int(self.int_type, i as i32),
                        )
                        .to_rvalue(),
                );
            }
            for i in 0..(mask_num_units - vec_num_units) {
                elements.push(
                    self.context
                        .new_vector_access(
                            self.location,
                            v2,
                            self.context.new_rvalue_from_int(self.int_type, i as i32),
                        )
                        .to_rvalue(),
                );
            }
            let v1 = self.context.new_rvalue_from_vector(self.location, result_type, &elements);
            let zero = self.context.new_rvalue_zero(element_type);
            let v2 = self.context.new_rvalue_from_vector(
                self.location,
                result_type,
                &vec![zero; mask_num_units],
            );
            (v1, v2)
        } else {
            (v1, v2)
        };

        let new_mask_num_units = std::cmp::max(mask_num_units, vec_num_units);
        let mask_type = self.context.new_vector_type(mask_element_type, new_mask_num_units as u64);
        let mask = self.context.new_rvalue_from_vector(self.location, mask_type, &mask_elements);
        let result = self.context.new_rvalue_vector_perm(self.location, v1, v2, mask);

        if vec_num_units != mask_num_units {
            // NOTE: if padding was added, only select the number of elements of the masks to
            // remove that padding in the result.
            let mut elements = vec![];
            for i in 0..mask_num_units {
                elements.push(
                    self.context
                        .new_vector_access(
                            self.location,
                            result,
                            self.context.new_rvalue_from_int(self.int_type, i as i32),
                        )
                        .to_rvalue(),
                );
            }
            self.context.new_rvalue_from_vector(self.location, result_type, &elements)
        } else {
            result
        }
    }

    #[cfg(not(feature = "master"))]
    pub fn shuffle_vector(
        &mut self,
        _v1: RValue<'gcc>,
        _v2: RValue<'gcc>,
        _mask: RValue<'gcc>,
    ) -> RValue<'gcc> {
        unimplemented!();
    }

    #[cfg(feature = "master")]
    pub fn vector_reduce<F>(&mut self, src: RValue<'gcc>, op: F) -> RValue<'gcc>
    where
        F: Fn(RValue<'gcc>, RValue<'gcc>, &'gcc Context<'gcc>) -> RValue<'gcc>,
    {
        let vector_type = src.get_type().unqualified().dyncast_vector().expect("vector type");
        let element_type = vector_type.get_element_type();
        let mask_element_type = self.type_ix(element_type.get_size() as u64 * 8);
        let element_count = vector_type.get_num_units();
        let mut vector_elements = vec![];
        for i in 0..element_count {
            vector_elements.push(i);
        }
        let mask_type = self.context.new_vector_type(mask_element_type, element_count as u64);
        let mut shift = 1;
        let mut res = src;
        while shift < element_count {
            let vector_elements: Vec<_> = vector_elements
                .iter()
                .map(|i| {
                    self.context.new_rvalue_from_int(
                        mask_element_type,
                        ((i + shift) % element_count) as i32,
                    )
                })
                .collect();
            let mask =
                self.context.new_rvalue_from_vector(self.location, mask_type, &vector_elements);
            let shifted = self.context.new_rvalue_vector_perm(self.location, res, res, mask);
            shift *= 2;
            res = op(res, shifted, self.context);
        }
        self.context
            .new_vector_access(self.location, res, self.context.new_rvalue_zero(self.int_type))
            .to_rvalue()
    }

    #[cfg(not(feature = "master"))]
    pub fn vector_reduce<F>(&mut self, _src: RValue<'gcc>, _op: F) -> RValue<'gcc>
    where
        F: Fn(RValue<'gcc>, RValue<'gcc>, &'gcc Context<'gcc>) -> RValue<'gcc>,
    {
        unimplemented!();
    }

    pub fn vector_reduce_op(&mut self, src: RValue<'gcc>, op: BinaryOp) -> RValue<'gcc> {
        let loc = self.location;
        self.vector_reduce(src, |a, b, context| context.new_binary_op(loc, op, a.get_type(), a, b))
    }

    pub fn vector_reduce_fadd_reassoc(
        &mut self,
        _acc: RValue<'gcc>,
        _src: RValue<'gcc>,
    ) -> RValue<'gcc> {
        unimplemented!();
    }

    #[cfg(feature = "master")]
    pub fn vector_reduce_fadd(&mut self, acc: RValue<'gcc>, src: RValue<'gcc>) -> RValue<'gcc> {
        let vector_type = src.get_type().unqualified().dyncast_vector().expect("vector type");
        let element_count = vector_type.get_num_units();
        (0..element_count)
            .map(|i| {
                self.context
                    .new_vector_access(
                        self.location,
                        src,
                        self.context.new_rvalue_from_int(self.int_type, i as _),
                    )
                    .to_rvalue()
            })
            .fold(acc, |x, i| x + i)
    }

    #[cfg(not(feature = "master"))]
    pub fn vector_reduce_fadd(&mut self, _acc: RValue<'gcc>, _src: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    pub fn vector_reduce_fmul_reassoc(
        &mut self,
        _acc: RValue<'gcc>,
        _src: RValue<'gcc>,
    ) -> RValue<'gcc> {
        unimplemented!();
    }

    #[cfg(feature = "master")]
    pub fn vector_reduce_fmul(&mut self, acc: RValue<'gcc>, src: RValue<'gcc>) -> RValue<'gcc> {
        let vector_type = src.get_type().unqualified().dyncast_vector().expect("vector type");
        let element_count = vector_type.get_num_units();
        (0..element_count)
            .map(|i| {
                self.context
                    .new_vector_access(
                        self.location,
                        src,
                        self.context.new_rvalue_from_int(self.int_type, i as _),
                    )
                    .to_rvalue()
            })
            .fold(acc, |x, i| x * i)
    }

    #[cfg(not(feature = "master"))]
    pub fn vector_reduce_fmul(&mut self, _acc: RValue<'gcc>, _src: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!()
    }

    // Inspired by Hacker's Delight min implementation.
    pub fn vector_reduce_min(&mut self, src: RValue<'gcc>) -> RValue<'gcc> {
        let loc = self.location;
        self.vector_reduce(src, |a, b, context| {
            let differences_or_zeros = difference_or_zero(loc, a, b, context);
            context.new_binary_op(loc, BinaryOp::Plus, b.get_type(), b, differences_or_zeros)
        })
    }

    // Inspired by Hacker's Delight max implementation.
    pub fn vector_reduce_max(&mut self, src: RValue<'gcc>) -> RValue<'gcc> {
        let loc = self.location;
        self.vector_reduce(src, |a, b, context| {
            let differences_or_zeros = difference_or_zero(loc, a, b, context);
            context.new_binary_op(loc, BinaryOp::Minus, a.get_type(), a, differences_or_zeros)
        })
    }

    fn vector_extremum(
        &mut self,
        a: RValue<'gcc>,
        b: RValue<'gcc>,
        direction: ExtremumOperation,
    ) -> RValue<'gcc> {
        let vector_type = a.get_type();

        // mask out the NaNs in b and replace them with the corresponding lane in a, so when a and
        // b get compared & spliced together, we get the numeric values instead of NaNs.
        let b_nan_mask = self.context.new_comparison(self.location, ComparisonOp::NotEquals, b, b);
        let mask_type = b_nan_mask.get_type();
        let b_nan_mask_inverted =
            self.context.new_unary_op(self.location, UnaryOp::BitwiseNegate, mask_type, b_nan_mask);
        let a_cast = self.context.new_bitcast(self.location, a, mask_type);
        let b_cast = self.context.new_bitcast(self.location, b, mask_type);
        let res = (b_nan_mask & a_cast) | (b_nan_mask_inverted & b_cast);
        let b = self.context.new_bitcast(self.location, res, vector_type);

        // now do the actual comparison
        let comparison_op = match direction {
            ExtremumOperation::Min => ComparisonOp::LessThan,
            ExtremumOperation::Max => ComparisonOp::GreaterThan,
        };
        let cmp = self.context.new_comparison(self.location, comparison_op, a, b);
        let cmp_inverted =
            self.context.new_unary_op(self.location, UnaryOp::BitwiseNegate, cmp.get_type(), cmp);
        let res = (cmp & a_cast) | (cmp_inverted & res);
        self.context.new_bitcast(self.location, res, vector_type)
    }

    pub fn vector_fmin(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.vector_extremum(a, b, ExtremumOperation::Min)
    }

    #[cfg(feature = "master")]
    pub fn vector_reduce_fmin(&mut self, src: RValue<'gcc>) -> RValue<'gcc> {
        let vector_type = src.get_type().unqualified().dyncast_vector().expect("vector type");
        let element_count = vector_type.get_num_units();
        let mut acc = self
            .context
            .new_vector_access(self.location, src, self.context.new_rvalue_zero(self.int_type))
            .to_rvalue();
        for i in 1..element_count {
            let elem = self
                .context
                .new_vector_access(
                    self.location,
                    src,
                    self.context.new_rvalue_from_int(self.int_type, i as _),
                )
                .to_rvalue();
            let cmp = self.context.new_comparison(self.location, ComparisonOp::LessThan, acc, elem);
            acc = self.select(cmp, acc, elem);
        }
        acc
    }

    #[cfg(not(feature = "master"))]
    pub fn vector_reduce_fmin(&mut self, _src: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    pub fn vector_fmax(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.vector_extremum(a, b, ExtremumOperation::Max)
    }

    #[cfg(feature = "master")]
    pub fn vector_reduce_fmax(&mut self, src: RValue<'gcc>) -> RValue<'gcc> {
        let vector_type = src.get_type().unqualified().dyncast_vector().expect("vector type");
        let element_count = vector_type.get_num_units();
        let mut acc = self
            .context
            .new_vector_access(self.location, src, self.context.new_rvalue_zero(self.int_type))
            .to_rvalue();
        for i in 1..element_count {
            let elem = self
                .context
                .new_vector_access(
                    self.location,
                    src,
                    self.context.new_rvalue_from_int(self.int_type, i as _),
                )
                .to_rvalue();
            let cmp =
                self.context.new_comparison(self.location, ComparisonOp::GreaterThan, acc, elem);
            acc = self.select(cmp, acc, elem);
        }
        acc
    }

    #[cfg(not(feature = "master"))]
    pub fn vector_reduce_fmax(&mut self, _src: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    pub fn vector_select(
        &mut self,
        cond: RValue<'gcc>,
        then_val: RValue<'gcc>,
        else_val: RValue<'gcc>,
    ) -> RValue<'gcc> {
        // cond is a vector of integers, not of bools.
        let vector_type = cond.get_type().unqualified().dyncast_vector().expect("vector type");
        let num_units = vector_type.get_num_units();
        let element_type = vector_type.get_element_type();

        #[cfg(feature = "master")]
        let (cond, element_type) = {
            // TODO(antoyo): dyncast_vector should not require a call to unqualified.
            let then_val_vector_type =
                then_val.get_type().unqualified().dyncast_vector().expect("vector type");
            let then_val_element_type = then_val_vector_type.get_element_type();
            let then_val_element_size = then_val_element_type.get_size();

            // NOTE: the mask needs to be of the same size as the other arguments in order for the &
            // operation to work.
            if then_val_element_size != element_type.get_size() {
                let new_element_type = self.type_ix(then_val_element_size as u64 * 8);
                let new_vector_type =
                    self.context.new_vector_type(new_element_type, num_units as u64);
                let cond = self.context.convert_vector(self.location, cond, new_vector_type);
                (cond, new_element_type)
            } else {
                (cond, element_type)
            }
        };

        let cond_type = cond.get_type();

        let zeros = vec![self.context.new_rvalue_zero(element_type); num_units];
        let zeros = self.context.new_rvalue_from_vector(self.location, cond_type, &zeros);

        let result_type = then_val.get_type();

        let masks =
            self.context.new_comparison(self.location, ComparisonOp::NotEquals, cond, zeros);
        // NOTE: masks is a vector of integers, but the values can be vectors of floats, so use bitcast to make
        // the & operation work.
        let then_val = self.bitcast_if_needed(then_val, masks.get_type());
        let then_vals = masks & then_val;

        let minus_ones = vec![self.context.new_rvalue_from_int(element_type, -1); num_units];
        let minus_ones = self.context.new_rvalue_from_vector(self.location, cond_type, &minus_ones);
        let inverted_masks = masks ^ minus_ones;
        // NOTE: sometimes, the type of else_val can be different than the type of then_val in
        // libgccjit (vector of int vs vector of int32_t), but they should be the same for the AND
        // operation to work.
        // TODO: remove bitcast now that vector types can be compared?
        let else_val = self.context.new_bitcast(self.location, else_val, then_val.get_type());
        let else_vals = inverted_masks & else_val;

        let res = then_vals | else_vals;
        self.bitcast_if_needed(res, result_type)
    }
}

fn difference_or_zero<'gcc>(
    loc: Option<Location<'gcc>>,
    a: RValue<'gcc>,
    b: RValue<'gcc>,
    context: &'gcc Context<'gcc>,
) -> RValue<'gcc> {
    let difference = a - b;
    let masks = context.new_comparison(loc, ComparisonOp::GreaterThanEquals, b, a);
    // NOTE: masks is a vector of integers, but the values can be vectors of floats, so use bitcast to make
    // the & operation work.
    let a_type = a.get_type();
    let masks =
        if masks.get_type() != a_type { context.new_bitcast(loc, masks, a_type) } else { masks };
    difference & masks
}

impl<'a, 'gcc, 'tcx> StaticBuilderMethods for Builder<'a, 'gcc, 'tcx> {
    fn get_static(&mut self, def_id: DefId) -> RValue<'gcc> {
        // Forward to the `get_static` method of `CodegenCx`
        self.cx().get_static(def_id).get_address(self.location)
    }
}

impl<'tcx> HasTypingEnv<'tcx> for Builder<'_, '_, 'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.cx.typing_env()
    }
}

impl<'tcx> HasTargetSpec for Builder<'_, '_, 'tcx> {
    fn target_spec(&self) -> &Target {
        self.cx.target_spec()
    }
}

impl<'tcx> HasX86AbiOpt for Builder<'_, '_, 'tcx> {
    fn x86_abi_opt(&self) -> X86Abi {
        self.cx.x86_abi_opt()
    }
}

pub trait ToGccComp {
    fn to_gcc_comparison(&self) -> ComparisonOp;
}

impl ToGccComp for IntPredicate {
    fn to_gcc_comparison(&self) -> ComparisonOp {
        match *self {
            IntPredicate::IntEQ => ComparisonOp::Equals,
            IntPredicate::IntNE => ComparisonOp::NotEquals,
            IntPredicate::IntUGT => ComparisonOp::GreaterThan,
            IntPredicate::IntUGE => ComparisonOp::GreaterThanEquals,
            IntPredicate::IntULT => ComparisonOp::LessThan,
            IntPredicate::IntULE => ComparisonOp::LessThanEquals,
            IntPredicate::IntSGT => ComparisonOp::GreaterThan,
            IntPredicate::IntSGE => ComparisonOp::GreaterThanEquals,
            IntPredicate::IntSLT => ComparisonOp::LessThan,
            IntPredicate::IntSLE => ComparisonOp::LessThanEquals,
        }
    }
}

impl ToGccComp for RealPredicate {
    fn to_gcc_comparison(&self) -> ComparisonOp {
        // TODO(antoyo): check that ordered vs non-ordered is respected.
        match *self {
            RealPredicate::RealPredicateFalse => unreachable!(),
            RealPredicate::RealOEQ => ComparisonOp::Equals,
            RealPredicate::RealOGT => ComparisonOp::GreaterThan,
            RealPredicate::RealOGE => ComparisonOp::GreaterThanEquals,
            RealPredicate::RealOLT => ComparisonOp::LessThan,
            RealPredicate::RealOLE => ComparisonOp::LessThanEquals,
            RealPredicate::RealONE => ComparisonOp::NotEquals,
            RealPredicate::RealORD => unreachable!(),
            RealPredicate::RealUNO => unreachable!(),
            RealPredicate::RealUEQ => ComparisonOp::Equals,
            RealPredicate::RealUGT => ComparisonOp::GreaterThan,
            RealPredicate::RealUGE => ComparisonOp::GreaterThan,
            RealPredicate::RealULT => ComparisonOp::LessThan,
            RealPredicate::RealULE => ComparisonOp::LessThan,
            RealPredicate::RealUNE => ComparisonOp::NotEquals,
            RealPredicate::RealPredicateTrue => unreachable!(),
        }
    }
}

#[repr(C)]
#[allow(non_camel_case_types)]
enum MemOrdering {
    __ATOMIC_RELAXED,
    __ATOMIC_CONSUME,
    __ATOMIC_ACQUIRE,
    __ATOMIC_RELEASE,
    __ATOMIC_ACQ_REL,
    __ATOMIC_SEQ_CST,
}

trait ToGccOrdering {
    fn to_gcc(self) -> i32;
}

impl ToGccOrdering for AtomicOrdering {
    fn to_gcc(self) -> i32 {
        use MemOrdering::*;

        let ordering = match self {
            AtomicOrdering::Relaxed => __ATOMIC_RELAXED, // TODO(antoyo): check if that's the same.
            AtomicOrdering::Acquire => __ATOMIC_ACQUIRE,
            AtomicOrdering::Release => __ATOMIC_RELEASE,
            AtomicOrdering::AcqRel => __ATOMIC_ACQ_REL,
            AtomicOrdering::SeqCst => __ATOMIC_SEQ_CST,
        };
        ordering as i32
    }
}

// Needed because gcc 12 `get_size()` doesn't work on pointers.
#[cfg(feature = "master")]
fn get_maybe_pointer_size(value: RValue<'_>) -> u32 {
    value.get_type().get_size()
}

#[cfg(not(feature = "master"))]
fn get_maybe_pointer_size(value: RValue<'_>) -> u32 {
    let type_ = value.get_type();
    if type_.get_pointee().is_some() { size_of::<*const ()>() as _ } else { type_.get_size() }
}
