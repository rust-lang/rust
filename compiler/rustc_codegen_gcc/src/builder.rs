use std::borrow::Cow;
use std::cell::Cell;
use std::convert::TryFrom;
use std::ops::Deref;

use gccjit::FunctionType;
use gccjit::{
    BinaryOp,
    Block,
    ComparisonOp,
    Function,
    LValue,
    RValue,
    ToRValue,
    Type,
    UnaryOp,
};
use rustc_codegen_ssa::MemFlags;
use rustc_codegen_ssa::common::{AtomicOrdering, AtomicRmwBinOp, IntPredicate, RealPredicate, SynchronizationScope};
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::{
    BackendTypes,
    BaseTypeMethods,
    BuilderMethods,
    ConstMethods,
    DerivedTypeMethods,
    LayoutTypeMethods,
    HasCodegen,
    OverflowOp,
    StaticBuilderMethods,
};
use rustc_middle::ty::{ParamEnv, Ty, TyCtxt};
use rustc_middle::ty::layout::{FnAbiError, FnAbiOfHelpers, FnAbiRequest, HasParamEnv, HasTyCtxt, LayoutError, LayoutOfHelpers, TyAndLayout};
use rustc_span::Span;
use rustc_span::def_id::DefId;
use rustc_target::abi::{
    self,
    call::FnAbi,
    Align,
    HasDataLayout,
    Size,
    TargetDataLayout,
    WrappingRange,
};
use rustc_target::spec::{HasTargetSpec, Target};

use crate::common::{SignType, TypeReflection, type_is_pointer};
use crate::context::CodegenCx;
use crate::type_of::LayoutGccExt;

// TODO(antoyo)
type Funclet = ();

// TODO(antoyo): remove this variable.
static mut RETURN_VALUE_COUNT: usize = 0;

enum ExtremumOperation {
    Max,
    Min,
}

trait EnumClone {
    fn clone(&self) -> Self;
}

impl EnumClone for AtomicOrdering {
    fn clone(&self) -> Self {
        match *self {
            AtomicOrdering::NotAtomic => AtomicOrdering::NotAtomic,
            AtomicOrdering::Unordered => AtomicOrdering::Unordered,
            AtomicOrdering::Monotonic => AtomicOrdering::Monotonic,
            AtomicOrdering::Acquire => AtomicOrdering::Acquire,
            AtomicOrdering::Release => AtomicOrdering::Release,
            AtomicOrdering::AcquireRelease => AtomicOrdering::AcquireRelease,
            AtomicOrdering::SequentiallyConsistent => AtomicOrdering::SequentiallyConsistent,
        }
    }
}

pub struct Builder<'a: 'gcc, 'gcc, 'tcx> {
    pub cx: &'a CodegenCx<'gcc, 'tcx>,
    pub block: Option<Block<'gcc>>,
    stack_var_count: Cell<usize>,
}

impl<'a, 'gcc, 'tcx> Builder<'a, 'gcc, 'tcx> {
    fn with_cx(cx: &'a CodegenCx<'gcc, 'tcx>) -> Self {
        Builder {
            cx,
            block: None,
            stack_var_count: Cell::new(0),
        }
    }

    fn atomic_extremum(&mut self, operation: ExtremumOperation, dst: RValue<'gcc>, src: RValue<'gcc>, order: AtomicOrdering) -> RValue<'gcc> {
        let size = self.cx.int_width(src.get_type()) / 8;

        let func = self.current_func();

        let load_ordering =
            match order {
                // TODO(antoyo): does this make sense?
                AtomicOrdering::AcquireRelease | AtomicOrdering::Release => AtomicOrdering::Acquire,
                _ => order.clone(),
            };
        let previous_value = self.atomic_load(dst.get_type(), dst, load_ordering.clone(), Size::from_bytes(size));
        let previous_var = func.new_local(None, previous_value.get_type(), "previous_value");
        let return_value = func.new_local(None, previous_value.get_type(), "return_value");
        self.llbb().add_assignment(None, previous_var, previous_value);
        self.llbb().add_assignment(None, return_value, previous_var.to_rvalue());

        let while_block = func.new_block("while");
        let after_block = func.new_block("after_while");
        self.llbb().end_with_jump(None, while_block);

        // NOTE: since jumps were added and compare_exchange doesn't expect this, the current blocks in the
        // state need to be updated.
        self.block = Some(while_block);
        *self.cx.current_block.borrow_mut() = Some(while_block);

        let comparison_operator =
            match operation {
                ExtremumOperation::Max => ComparisonOp::LessThan,
                ExtremumOperation::Min => ComparisonOp::GreaterThan,
            };

        let cond1 = self.context.new_comparison(None, comparison_operator, previous_var.to_rvalue(), self.context.new_cast(None, src, previous_value.get_type()));
        let compare_exchange = self.compare_exchange(dst, previous_var, src, order, load_ordering, false);
        let cond2 = self.cx.context.new_unary_op(None, UnaryOp::LogicalNegate, compare_exchange.get_type(), compare_exchange);
        let cond = self.cx.context.new_binary_op(None, BinaryOp::LogicalAnd, self.cx.bool_type, cond1, cond2);

        while_block.end_with_conditional(None, cond, while_block, after_block);

        // NOTE: since jumps were added in a place rustc does not expect, the current blocks in the
        // state need to be updated.
        self.block = Some(after_block);
        *self.cx.current_block.borrow_mut() = Some(after_block);

        return_value.to_rvalue()
    }

    fn compare_exchange(&self, dst: RValue<'gcc>, cmp: LValue<'gcc>, src: RValue<'gcc>, order: AtomicOrdering, failure_order: AtomicOrdering, weak: bool) -> RValue<'gcc> {
        let size = self.cx.int_width(src.get_type());
        let compare_exchange = self.context.get_builtin_function(&format!("__atomic_compare_exchange_{}", size / 8));
        let order = self.context.new_rvalue_from_int(self.i32_type, order.to_gcc());
        let failure_order = self.context.new_rvalue_from_int(self.i32_type, failure_order.to_gcc());
        let weak = self.context.new_rvalue_from_int(self.bool_type, weak as i32);

        let void_ptr_type = self.context.new_type::<*mut ()>();
        let volatile_void_ptr_type = void_ptr_type.make_volatile();
        let dst = self.context.new_cast(None, dst, volatile_void_ptr_type);
        let expected = self.context.new_cast(None, cmp.get_address(None), void_ptr_type);

        // NOTE: not sure why, but we have the wrong type here.
        let int_type = compare_exchange.get_param(2).to_rvalue().get_type();
        let src = self.context.new_cast(None, src, int_type);
        self.context.new_call(None, compare_exchange, &[dst, expected, src, weak, order, failure_order])
    }

    pub fn assign(&self, lvalue: LValue<'gcc>, value: RValue<'gcc>) {
        self.llbb().add_assignment(None, lvalue, value);
    }

    fn check_call<'b>(&mut self, _typ: &str, func: Function<'gcc>, args: &'b [RValue<'gcc>]) -> Cow<'b, [RValue<'gcc>]> {
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
            .enumerate()
            .map(|(_i, (expected_ty, &actual_val))| {
                let actual_ty = actual_val.get_type();
                if expected_ty != actual_ty {
                    self.bitcast(actual_val, expected_ty)
                }
                else {
                    actual_val
                }
            })
            .collect();

        Cow::Owned(casted_args)
    }

    fn check_ptr_call<'b>(&mut self, _typ: &str, func_ptr: RValue<'gcc>, args: &'b [RValue<'gcc>]) -> Cow<'b, [RValue<'gcc>]> {
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

        if all_args_match {
            return Cow::Borrowed(args);
        }

        let casted_args: Vec<_> = param_types
            .into_iter()
            .zip(args.iter())
            .enumerate()
            .map(|(_i, (expected_ty, &actual_val))| {
                let actual_ty = actual_val.get_type();
                if expected_ty != actual_ty {
                    self.bitcast(actual_val, expected_ty)
                }
                else {
                    actual_val
                }
            })
            .collect();

        Cow::Owned(casted_args)
    }

    fn check_store(&mut self, val: RValue<'gcc>, ptr: RValue<'gcc>) -> RValue<'gcc> {
        let dest_ptr_ty = self.cx.val_ty(ptr).make_pointer(); // TODO(antoyo): make sure make_pointer() is okay here.
        let stored_ty = self.cx.val_ty(val);
        let stored_ptr_ty = self.cx.type_ptr_to(stored_ty);

        if dest_ptr_ty == stored_ptr_ty {
            ptr
        }
        else {
            self.bitcast(ptr, stored_ptr_ty)
        }
    }

    pub fn current_func(&self) -> Function<'gcc> {
        self.block.expect("block").get_function()
    }

    fn function_call(&mut self, func: RValue<'gcc>, args: &[RValue<'gcc>], _funclet: Option<&Funclet>) -> RValue<'gcc> {
        // TODO(antoyo): remove when the API supports a different type for functions.
        let func: Function<'gcc> = self.cx.rvalue_as_function(func);
        let args = self.check_call("call", func, args);

        // gccjit requires to use the result of functions, even when it's not used.
        // That's why we assign the result to a local or call add_eval().
        let return_type = func.get_return_type();
        let current_block = self.current_block.borrow().expect("block");
        let void_type = self.context.new_type::<()>();
        let current_func = current_block.get_function();
        if return_type != void_type {
            unsafe { RETURN_VALUE_COUNT += 1 };
            let result = current_func.new_local(None, return_type, &format!("returnValue{}", unsafe { RETURN_VALUE_COUNT }));
            current_block.add_assignment(None, result, self.cx.context.new_call(None, func, &args));
            result.to_rvalue()
        }
        else {
            current_block.add_eval(None, self.cx.context.new_call(None, func, &args));
            // Return dummy value when not having return value.
            self.context.new_rvalue_from_long(self.isize_type, 0)
        }
    }

    fn function_ptr_call(&mut self, func_ptr: RValue<'gcc>, args: &[RValue<'gcc>], _funclet: Option<&Funclet>) -> RValue<'gcc> {
        let args = self.check_ptr_call("call", func_ptr, args);

        // gccjit requires to use the result of functions, even when it's not used.
        // That's why we assign the result to a local or call add_eval().
        let gcc_func = func_ptr.get_type().dyncast_function_ptr_type().expect("function ptr");
        let mut return_type = gcc_func.get_return_type();
        let current_block = self.current_block.borrow().expect("block");
        let void_type = self.context.new_type::<()>();
        let current_func = current_block.get_function();

        // FIXME(antoyo): As a temporary workaround for unsupported LLVM intrinsics.
        if gcc_func.get_param_count() == 0 && format!("{:?}", func_ptr) == "__builtin_ia32_pmovmskb128" {
            return_type = self.int_type;
        }

        if return_type != void_type {
            unsafe { RETURN_VALUE_COUNT += 1 };
            let result = current_func.new_local(None, return_type, &format!("returnValue{}", unsafe { RETURN_VALUE_COUNT }));
            current_block.add_assignment(None, result, self.cx.context.new_call_through_ptr(None, func_ptr, &args));
            result.to_rvalue()
        }
        else {
            if gcc_func.get_param_count() == 0 {
                // FIXME(antoyo): As a temporary workaround for unsupported LLVM intrinsics.
                current_block.add_eval(None, self.cx.context.new_call_through_ptr(None, func_ptr, &[]));
            }
            else {
                current_block.add_eval(None, self.cx.context.new_call_through_ptr(None, func_ptr, &args));
            }
            // Return dummy value when not having return value.
            let result = current_func.new_local(None, self.isize_type, "dummyValueThatShouldNeverBeUsed");
            current_block.add_assignment(None, result, self.context.new_rvalue_from_long(self.isize_type, 0));
            result.to_rvalue()
        }
    }

    pub fn overflow_call(&mut self, func: Function<'gcc>, args: &[RValue<'gcc>], _funclet: Option<&Funclet>) -> RValue<'gcc> {
        // gccjit requires to use the result of functions, even when it's not used.
        // That's why we assign the result to a local.
        let return_type = self.context.new_type::<bool>();
        let current_block = self.current_block.borrow().expect("block");
        let current_func = current_block.get_function();
        // TODO(antoyo): return the new_call() directly? Since the overflow function has no side-effects.
        unsafe { RETURN_VALUE_COUNT += 1 };
        let result = current_func.new_local(None, return_type, &format!("returnValue{}", unsafe { RETURN_VALUE_COUNT }));
        current_block.add_assignment(None, result, self.cx.context.new_call(None, func, &args));
        result.to_rvalue()
    }
}

impl<'gcc, 'tcx> HasCodegen<'tcx> for Builder<'_, 'gcc, 'tcx> {
    type CodegenCx = CodegenCx<'gcc, 'tcx>;
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
    type LayoutOfResult = TyAndLayout<'tcx>;

    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        self.cx.handle_layout_err(err, span, ty)
    }
}

impl<'tcx> FnAbiOfHelpers<'tcx> for Builder<'_, '_, 'tcx> {
    type FnAbiOfResult = &'tcx FnAbi<'tcx, Ty<'tcx>>;

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

impl<'gcc, 'tcx> Deref for Builder<'_, 'gcc, 'tcx> {
    type Target = CodegenCx<'gcc, 'tcx>;

    fn deref(&self) -> &Self::Target {
        self.cx
    }
}

impl<'gcc, 'tcx> BackendTypes for Builder<'_, 'gcc, 'tcx> {
    type Value = <CodegenCx<'gcc, 'tcx> as BackendTypes>::Value;
    type Function = <CodegenCx<'gcc, 'tcx> as BackendTypes>::Function;
    type BasicBlock = <CodegenCx<'gcc, 'tcx> as BackendTypes>::BasicBlock;
    type Type = <CodegenCx<'gcc, 'tcx> as BackendTypes>::Type;
    type Funclet = <CodegenCx<'gcc, 'tcx> as BackendTypes>::Funclet;

    type DIScope = <CodegenCx<'gcc, 'tcx> as BackendTypes>::DIScope;
    type DILocation = <CodegenCx<'gcc, 'tcx> as BackendTypes>::DILocation;
    type DIVariable = <CodegenCx<'gcc, 'tcx> as BackendTypes>::DIVariable;
}

impl<'a, 'gcc, 'tcx> BuilderMethods<'a, 'tcx> for Builder<'a, 'gcc, 'tcx> {
    fn build(cx: &'a CodegenCx<'gcc, 'tcx>, block: Block<'gcc>) -> Self {
        let mut bx = Builder::with_cx(cx);
        *cx.current_block.borrow_mut() = Some(block);
        bx.block = Some(block);
        bx
    }

    fn build_sibling_block(&mut self, name: &str) -> Self {
        let block = self.append_sibling_block(name);
        Self::build(self.cx, block)
    }

    fn llbb(&self) -> Block<'gcc> {
        self.block.expect("block")
    }

    fn append_block(cx: &'a CodegenCx<'gcc, 'tcx>, func: RValue<'gcc>, name: &str) -> Block<'gcc> {
        let func = cx.rvalue_as_function(func);
        func.new_block(name)
    }

    fn append_sibling_block(&mut self, name: &str) -> Block<'gcc> {
        let func = self.current_func();
        func.new_block(name)
    }

    fn ret_void(&mut self) {
        self.llbb().end_with_void_return(None)
    }

    fn ret(&mut self, value: RValue<'gcc>) {
        let value =
            if self.structs_as_pointer.borrow().contains(&value) {
                // NOTE: hack to workaround a limitation of the rustc API: see comment on
                // CodegenCx.structs_as_pointer
                value.dereference(None).to_rvalue()
            }
            else {
                value
            };
        self.llbb().end_with_return(None, value);
    }

    fn br(&mut self, dest: Block<'gcc>) {
        self.llbb().end_with_jump(None, dest)
    }

    fn cond_br(&mut self, cond: RValue<'gcc>, then_block: Block<'gcc>, else_block: Block<'gcc>) {
        self.llbb().end_with_conditional(None, cond, then_block, else_block)
    }

    fn switch(&mut self, value: RValue<'gcc>, default_block: Block<'gcc>, cases: impl ExactSizeIterator<Item = (u128, Block<'gcc>)>) {
        let mut gcc_cases = vec![];
        let typ = self.val_ty(value);
        for (on_val, dest) in cases {
            let on_val = self.const_uint_big(typ, on_val);
            gcc_cases.push(self.context.new_case(on_val, on_val, dest));
        }
        self.block.expect("block").end_with_switch(None, value, default_block, &gcc_cases);
    }

    fn invoke(&mut self, _typ: Type<'gcc>, _func: RValue<'gcc>, _args: &[RValue<'gcc>], then: Block<'gcc>, catch: Block<'gcc>, _funclet: Option<&Funclet>) -> RValue<'gcc> {
        let condition = self.context.new_rvalue_from_int(self.bool_type, 0);
        self.llbb().end_with_conditional(None, condition, then, catch);
        self.context.new_rvalue_from_int(self.int_type, 0)

        // TODO(antoyo)
    }

    fn unreachable(&mut self) {
        let func = self.context.get_builtin_function("__builtin_unreachable");
        let block = self.block.expect("block");
        block.add_eval(None, self.context.new_call(None, func, &[]));
        let return_type = block.get_function().get_return_type();
        let void_type = self.context.new_type::<()>();
        if return_type == void_type {
            block.end_with_void_return(None)
        }
        else {
            let return_value = self.current_func()
                .new_local(None, return_type, "unreachableReturn");
            block.end_with_return(None, return_value)
        }
    }

    fn add(&mut self, a: RValue<'gcc>, mut b: RValue<'gcc>) -> RValue<'gcc> {
        // FIXME(antoyo): this should not be required.
        if format!("{:?}", a.get_type()) != format!("{:?}", b.get_type()) {
            b = self.context.new_cast(None, b, a.get_type());
        }
        a + b
    }

    fn fadd(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a + b
    }

    fn sub(&mut self, a: RValue<'gcc>, mut b: RValue<'gcc>) -> RValue<'gcc> {
        if a.get_type() != b.get_type() {
            b = self.context.new_cast(None, b, a.get_type());
        }
        a - b
    }

    fn fsub(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a - b
    }

    fn mul(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a * b
    }

    fn fmul(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a * b
    }

    fn udiv(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): convert the arguments to unsigned?
        a / b
    }

    fn exactudiv(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): convert the arguments to unsigned?
        // TODO(antoyo): poison if not exact.
        a / b
    }

    fn sdiv(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): convert the arguments to signed?
        a / b
    }

    fn exactsdiv(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): posion if not exact.
        // FIXME(antoyo): rustc_codegen_ssa::mir::intrinsic uses different types for a and b but they
        // should be the same.
        let typ = a.get_type().to_signed(self);
        let b = self.context.new_cast(None, b, typ);
        a / b
    }

    fn fdiv(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a / b
    }

    fn urem(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a % b
    }

    fn srem(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a % b
    }

    fn frem(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        if a.get_type() == self.cx.float_type {
            let fmodf = self.context.get_builtin_function("fmodf");
            // FIXME(antoyo): this seems to produce the wrong result.
            return self.context.new_call(None, fmodf, &[a, b]);
        }
        assert_eq!(a.get_type(), self.cx.double_type);

        let fmod = self.context.get_builtin_function("fmod");
        return self.context.new_call(None, fmod, &[a, b]);
    }

    fn shl(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // FIXME(antoyo): remove the casts when libgccjit can shift an unsigned number by an unsigned number.
        let a_type = a.get_type();
        let b_type = b.get_type();
        if a_type.is_unsigned(self) && b_type.is_signed(self) {
            let a = self.context.new_cast(None, a, b_type);
            let result = a << b;
            self.context.new_cast(None, result, a_type)
        }
        else if a_type.is_signed(self) && b_type.is_unsigned(self) {
            let b = self.context.new_cast(None, b, a_type);
            a << b
        }
        else {
            a << b
        }
    }

    fn lshr(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // FIXME(antoyo): remove the casts when libgccjit can shift an unsigned number by an unsigned number.
        // TODO(antoyo): cast to unsigned to do a logical shift if that does not work.
        let a_type = a.get_type();
        let b_type = b.get_type();
        if a_type.is_unsigned(self) && b_type.is_signed(self) {
            let a = self.context.new_cast(None, a, b_type);
            let result = a >> b;
            self.context.new_cast(None, result, a_type)
        }
        else if a_type.is_signed(self) && b_type.is_unsigned(self) {
            let b = self.context.new_cast(None, b, a_type);
            a >> b
        }
        else {
            a >> b
        }
    }

    fn ashr(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): check whether behavior is an arithmetic shift for >> .
        // FIXME(antoyo): remove the casts when libgccjit can shift an unsigned number by an unsigned number.
        let a_type = a.get_type();
        let b_type = b.get_type();
        if a_type.is_unsigned(self) && b_type.is_signed(self) {
            let a = self.context.new_cast(None, a, b_type);
            let result = a >> b;
            self.context.new_cast(None, result, a_type)
        }
        else if a_type.is_signed(self) && b_type.is_unsigned(self) {
            let b = self.context.new_cast(None, b, a_type);
            a >> b
        }
        else {
            a >> b
        }
    }

    fn and(&mut self, a: RValue<'gcc>, mut b: RValue<'gcc>) -> RValue<'gcc> {
        if a.get_type() != b.get_type() {
            b = self.context.new_cast(None, b, a.get_type());
        }
        a & b
    }

    fn or(&mut self, a: RValue<'gcc>, mut b: RValue<'gcc>) -> RValue<'gcc> {
        if a.get_type() != b.get_type() {
            b = self.context.new_cast(None, b, a.get_type());
        }
        a | b
    }

    fn xor(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a ^ b
    }

    fn neg(&mut self, a: RValue<'gcc>) -> RValue<'gcc> {
        self.cx.context.new_unary_op(None, UnaryOp::Minus, a.get_type(), a)
    }

    fn fneg(&mut self, a: RValue<'gcc>) -> RValue<'gcc> {
        self.cx.context.new_unary_op(None, UnaryOp::Minus, a.get_type(), a)
    }

    fn not(&mut self, a: RValue<'gcc>) -> RValue<'gcc> {
        let operation =
            if a.get_type().is_bool() {
                UnaryOp::LogicalNegate
            }
            else {
                UnaryOp::BitwiseNegate
            };
        self.cx.context.new_unary_op(None, operation, a.get_type(), a)
    }

    fn unchecked_sadd(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a + b
    }

    fn unchecked_uadd(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a + b
    }

    fn unchecked_ssub(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a - b
    }

    fn unchecked_usub(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): should generate poison value?
        a - b
    }

    fn unchecked_smul(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a * b
    }

    fn unchecked_umul(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        a * b
    }

    fn fadd_fast(&mut self, _lhs: RValue<'gcc>, _rhs: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    fn fsub_fast(&mut self, _lhs: RValue<'gcc>, _rhs: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    fn fmul_fast(&mut self, _lhs: RValue<'gcc>, _rhs: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    fn fdiv_fast(&mut self, _lhs: RValue<'gcc>, _rhs: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    fn frem_fast(&mut self, _lhs: RValue<'gcc>, _rhs: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    fn checked_binop(&mut self, oop: OverflowOp, typ: Ty<'_>, lhs: Self::Value, rhs: Self::Value) -> (Self::Value, Self::Value) {
        use rustc_middle::ty::{Int, IntTy::*, Uint, UintTy::*};

        let new_kind =
            match typ.kind() {
                Int(t @ Isize) => Int(t.normalize(self.tcx.sess.target.pointer_width)),
                Uint(t @ Usize) => Uint(t.normalize(self.tcx.sess.target.pointer_width)),
                t @ (Uint(_) | Int(_)) => t.clone(),
                _ => panic!("tried to get overflow intrinsic for op applied to non-int type"),
            };

        // TODO(antoyo): remove duplication with intrinsic?
        let name =
            match oop {
                OverflowOp::Add =>
                    match new_kind {
                        Int(I8) => "__builtin_add_overflow",
                        Int(I16) => "__builtin_add_overflow",
                        Int(I32) => "__builtin_sadd_overflow",
                        Int(I64) => "__builtin_saddll_overflow",
                        Int(I128) => "__builtin_add_overflow",

                        Uint(U8) => "__builtin_add_overflow",
                        Uint(U16) => "__builtin_add_overflow",
                        Uint(U32) => "__builtin_uadd_overflow",
                        Uint(U64) => "__builtin_uaddll_overflow",
                        Uint(U128) => "__builtin_add_overflow",

                        _ => unreachable!(),
                    },
                OverflowOp::Sub =>
                    match new_kind {
                        Int(I8) => "__builtin_sub_overflow",
                        Int(I16) => "__builtin_sub_overflow",
                        Int(I32) => "__builtin_ssub_overflow",
                        Int(I64) => "__builtin_ssubll_overflow",
                        Int(I128) => "__builtin_sub_overflow",

                        Uint(U8) => "__builtin_sub_overflow",
                        Uint(U16) => "__builtin_sub_overflow",
                        Uint(U32) => "__builtin_usub_overflow",
                        Uint(U64) => "__builtin_usubll_overflow",
                        Uint(U128) => "__builtin_sub_overflow",

                        _ => unreachable!(),
                    },
                OverflowOp::Mul =>
                    match new_kind {
                        Int(I8) => "__builtin_mul_overflow",
                        Int(I16) => "__builtin_mul_overflow",
                        Int(I32) => "__builtin_smul_overflow",
                        Int(I64) => "__builtin_smulll_overflow",
                        Int(I128) => "__builtin_mul_overflow",

                        Uint(U8) => "__builtin_mul_overflow",
                        Uint(U16) => "__builtin_mul_overflow",
                        Uint(U32) => "__builtin_umul_overflow",
                        Uint(U64) => "__builtin_umulll_overflow",
                        Uint(U128) => "__builtin_mul_overflow",

                        _ => unreachable!(),
                    },
            };

        let intrinsic = self.context.get_builtin_function(&name);
        let res = self.current_func()
            // TODO(antoyo): is it correct to use rhs type instead of the parameter typ?
            .new_local(None, rhs.get_type(), "binopResult")
            .get_address(None);
        let overflow = self.overflow_call(intrinsic, &[lhs, rhs, res], None);
        (res.dereference(None).to_rvalue(), overflow)
    }

    fn alloca(&mut self, ty: Type<'gcc>, align: Align) -> RValue<'gcc> {
        // FIXME(antoyo): this check that we don't call get_aligned() a second time on a type.
        // Ideally, we shouldn't need to do this check.
        let aligned_type =
            if ty == self.cx.u128_type || ty == self.cx.i128_type {
                ty
            }
            else {
                ty.get_aligned(align.bytes())
            };
        // TODO(antoyo): It might be better to return a LValue, but fixing the rustc API is non-trivial.
        self.stack_var_count.set(self.stack_var_count.get() + 1);
        self.current_func().new_local(None, aligned_type, &format!("stack_var_{}", self.stack_var_count.get())).get_address(None)
    }

    fn dynamic_alloca(&mut self, _ty: Type<'gcc>, _align: Align) -> RValue<'gcc> {
        unimplemented!();
    }

    fn array_alloca(&mut self, _ty: Type<'gcc>, _len: RValue<'gcc>, _align: Align) -> RValue<'gcc> {
        unimplemented!();
    }

    fn load(&mut self, _ty: Type<'gcc>, ptr: RValue<'gcc>, _align: Align) -> RValue<'gcc> {
        // TODO(antoyo): use ty.
        let block = self.llbb();
        let function = block.get_function();
        // NOTE: instead of returning the dereference here, we have to assign it to a variable in
        // the current basic block. Otherwise, it could be used in another basic block, causing a
        // dereference after a drop, for instance.
        // TODO(antoyo): handle align.
        let deref = ptr.dereference(None).to_rvalue();
        let value_type = deref.get_type();
        unsafe { RETURN_VALUE_COUNT += 1 };
        let loaded_value = function.new_local(None, value_type, &format!("loadedValue{}", unsafe { RETURN_VALUE_COUNT }));
        block.add_assignment(None, loaded_value, deref);
        loaded_value.to_rvalue()
    }

    fn volatile_load(&mut self, _ty: Type<'gcc>, ptr: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): use ty.
        let ptr = self.context.new_cast(None, ptr, ptr.get_type().make_volatile());
        ptr.dereference(None).to_rvalue()
    }

    fn atomic_load(&mut self, _ty: Type<'gcc>, ptr: RValue<'gcc>, order: AtomicOrdering, size: Size) -> RValue<'gcc> {
        // TODO(antoyo): use ty.
        // TODO(antoyo): handle alignment.
        let atomic_load = self.context.get_builtin_function(&format!("__atomic_load_{}", size.bytes()));
        let ordering = self.context.new_rvalue_from_int(self.i32_type, order.to_gcc());

        let volatile_const_void_ptr_type = self.context.new_type::<()>()
            .make_const()
            .make_volatile()
            .make_pointer();
        let ptr = self.context.new_cast(None, ptr, volatile_const_void_ptr_type);
        self.context.new_call(None, atomic_load, &[ptr, ordering])
    }

    fn load_operand(&mut self, place: PlaceRef<'tcx, RValue<'gcc>>) -> OperandRef<'tcx, RValue<'gcc>> {
        assert_eq!(place.llextra.is_some(), place.layout.is_unsized());

        if place.layout.is_zst() {
            return OperandRef::new_zst(self, place.layout);
        }

        fn scalar_load_metadata<'a, 'gcc, 'tcx>(bx: &mut Builder<'a, 'gcc, 'tcx>, load: RValue<'gcc>, scalar: &abi::Scalar) {
            let vr = scalar.valid_range.clone();
            match scalar.value {
                abi::Int(..) => {
                    if !scalar.is_always_valid(bx) {
                        bx.range_metadata(load, scalar.valid_range);
                    }
                }
                abi::Pointer if vr.start < vr.end && !vr.contains(0) => {
                    bx.nonnull_metadata(load);
                }
                _ => {}
            }
        }

        let val =
            if let Some(llextra) = place.llextra {
                OperandValue::Ref(place.llval, Some(llextra), place.align)
            }
            else if place.layout.is_gcc_immediate() {
                let load = self.load(place.llval.get_type(), place.llval, place.align);
                if let abi::Abi::Scalar(ref scalar) = place.layout.abi {
                    scalar_load_metadata(self, load, scalar);
                }
                OperandValue::Immediate(self.to_immediate(load, place.layout))
            }
            else if let abi::Abi::ScalarPair(ref a, ref b) = place.layout.abi {
                let b_offset = a.value.size(self).align_to(b.value.align(self).abi);
                let pair_type = place.layout.gcc_type(self, false);

                let mut load = |i, scalar: &abi::Scalar, align| {
                    let llptr = self.struct_gep(pair_type, place.llval, i as u64);
                    let load = self.load(llptr.get_type(), llptr, align);
                    scalar_load_metadata(self, load, scalar);
                    if scalar.is_bool() { self.trunc(load, self.type_i1()) } else { load }
                };

                OperandValue::Pair(
                    load(0, a, place.align),
                    load(1, b, place.align.restrict_for_offset(b_offset)),
                )
            }
            else {
                OperandValue::Ref(place.llval, None, place.align)
            };

        OperandRef { val, layout: place.layout }
    }

    fn write_operand_repeatedly(mut self, cg_elem: OperandRef<'tcx, RValue<'gcc>>, count: u64, dest: PlaceRef<'tcx, RValue<'gcc>>) -> Self {
        let zero = self.const_usize(0);
        let count = self.const_usize(count);
        let start = dest.project_index(&mut self, zero).llval;
        let end = dest.project_index(&mut self, count).llval;

        let mut header_bx = self.build_sibling_block("repeat_loop_header");
        let mut body_bx = self.build_sibling_block("repeat_loop_body");
        let next_bx = self.build_sibling_block("repeat_loop_next");

        let ptr_type = start.get_type();
        let current = self.llbb().get_function().new_local(None, ptr_type, "loop_var");
        let current_val = current.to_rvalue();
        self.assign(current, start);

        self.br(header_bx.llbb());

        let keep_going = header_bx.icmp(IntPredicate::IntNE, current_val, end);
        header_bx.cond_br(keep_going, body_bx.llbb(), next_bx.llbb());

        let align = dest.align.restrict_for_offset(dest.layout.field(self.cx(), 0).size);
        cg_elem.val.store(&mut body_bx, PlaceRef::new_sized_aligned(current_val, cg_elem.layout, align));

        let next = body_bx.inbounds_gep(self.backend_type(cg_elem.layout), current.to_rvalue(), &[self.const_usize(1)]);
        body_bx.llbb().add_assignment(None, current, next);
        body_bx.br(header_bx.llbb());

        next_bx
    }

    fn range_metadata(&mut self, _load: RValue<'gcc>, _range: WrappingRange) {
        // TODO(antoyo)
    }

    fn nonnull_metadata(&mut self, _load: RValue<'gcc>) {
        // TODO(antoyo)
    }

    fn type_metadata(&mut self, _function: RValue<'gcc>, _typeid: String) {
        // Unsupported.
    }

    fn typeid_metadata(&mut self, _typeid: String) -> RValue<'gcc> {
        // Unsupported.
        self.context.new_rvalue_from_int(self.int_type, 0)
    }


    fn store(&mut self, val: RValue<'gcc>, ptr: RValue<'gcc>, align: Align) -> RValue<'gcc> {
        self.store_with_flags(val, ptr, align, MemFlags::empty())
    }

    fn store_with_flags(&mut self, val: RValue<'gcc>, ptr: RValue<'gcc>, _align: Align, _flags: MemFlags) -> RValue<'gcc> {
        let ptr = self.check_store(val, ptr);
        self.llbb().add_assignment(None, ptr.dereference(None), val);
        // TODO(antoyo): handle align and flags.
        // NOTE: dummy value here since it's never used. FIXME(antoyo): API should not return a value here?
        self.cx.context.new_rvalue_zero(self.type_i32())
    }

    fn atomic_store(&mut self, value: RValue<'gcc>, ptr: RValue<'gcc>, order: AtomicOrdering, size: Size) {
        // TODO(antoyo): handle alignment.
        let atomic_store = self.context.get_builtin_function(&format!("__atomic_store_{}", size.bytes()));
        let ordering = self.context.new_rvalue_from_int(self.i32_type, order.to_gcc());
        let volatile_const_void_ptr_type = self.context.new_type::<()>()
            .make_volatile()
            .make_pointer();
        let ptr = self.context.new_cast(None, ptr, volatile_const_void_ptr_type);

        // FIXME(antoyo): fix libgccjit to allow comparing an integer type with an aligned integer type because
        // the following cast is required to avoid this error:
        // gcc_jit_context_new_call: mismatching types for argument 2 of function "__atomic_store_4": assignment to param arg1 (type: int) from loadedValue3577 (type: unsigned int  __attribute__((aligned(4))))
        let int_type = atomic_store.get_param(1).to_rvalue().get_type();
        let value = self.context.new_cast(None, value, int_type);
        self.llbb()
            .add_eval(None, self.context.new_call(None, atomic_store, &[ptr, value, ordering]));
    }

    fn gep(&mut self, _typ: Type<'gcc>, ptr: RValue<'gcc>, indices: &[RValue<'gcc>]) -> RValue<'gcc> {
        let mut result = ptr;
        for index in indices {
            result = self.context.new_array_access(None, result, *index).get_address(None).to_rvalue();
        }
        result
    }

    fn inbounds_gep(&mut self, _typ: Type<'gcc>, ptr: RValue<'gcc>, indices: &[RValue<'gcc>]) -> RValue<'gcc> {
        // FIXME(antoyo): would be safer if doing the same thing (loop) as gep.
        // TODO(antoyo): specify inbounds somehow.
        match indices.len() {
            1 => {
                self.context.new_array_access(None, ptr, indices[0]).get_address(None)
            },
            2 => {
                let array = ptr.dereference(None); // TODO(antoyo): assert that first index is 0?
                self.context.new_array_access(None, array, indices[1]).get_address(None)
            },
            _ => unimplemented!(),
        }
    }

    fn struct_gep(&mut self, value_type: Type<'gcc>, ptr: RValue<'gcc>, idx: u64) -> RValue<'gcc> {
        // FIXME(antoyo): it would be better if the API only called this on struct, not on arrays.
        assert_eq!(idx as usize as u64, idx);
        let value = ptr.dereference(None).to_rvalue();

        if value_type.dyncast_array().is_some() {
            let index = self.context.new_rvalue_from_long(self.u64_type, i64::try_from(idx).expect("i64::try_from"));
            let element = self.context.new_array_access(None, value, index);
            element.get_address(None)
        }
        else if let Some(vector_type) = value_type.dyncast_vector() {
            let array_type = vector_type.get_element_type().make_pointer();
            let array = self.bitcast(ptr, array_type);
            let index = self.context.new_rvalue_from_long(self.u64_type, i64::try_from(idx).expect("i64::try_from"));
            let element = self.context.new_array_access(None, array, index);
            element.get_address(None)
        }
        else if let Some(struct_type) = value_type.is_struct() {
            ptr.dereference_field(None, struct_type.get_field(idx as i32)).get_address(None)
        }
        else {
            panic!("Unexpected type {:?}", value_type);
        }
    }

    /* Casts */
    fn trunc(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): check that it indeed truncate the value.
        self.context.new_cast(None, value, dest_ty)
    }

    fn sext(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): check that it indeed sign extend the value.
        if dest_ty.dyncast_vector().is_some() {
            // TODO(antoyo): nothing to do as it is only for LLVM?
            return value;
        }
        self.context.new_cast(None, value, dest_ty)
    }

    fn fptoui(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        self.context.new_cast(None, value, dest_ty)
    }

    fn fptosi(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        self.context.new_cast(None, value, dest_ty)
    }

    fn uitofp(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        self.context.new_cast(None, value, dest_ty)
    }

    fn sitofp(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        self.context.new_cast(None, value, dest_ty)
    }

    fn fptrunc(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): make sure it truncates.
        self.context.new_cast(None, value, dest_ty)
    }

    fn fpext(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        self.context.new_cast(None, value, dest_ty)
    }

    fn ptrtoint(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        self.cx.ptrtoint(self.block.expect("block"), value, dest_ty)
    }

    fn inttoptr(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        self.cx.inttoptr(self.block.expect("block"), value, dest_ty)
    }

    fn bitcast(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        self.cx.const_bitcast(value, dest_ty)
    }

    fn intcast(&mut self, value: RValue<'gcc>, dest_typ: Type<'gcc>, _is_signed: bool) -> RValue<'gcc> {
        // NOTE: is_signed is for value, not dest_typ.
        self.cx.context.new_cast(None, value, dest_typ)
    }

    fn pointercast(&mut self, value: RValue<'gcc>, dest_ty: Type<'gcc>) -> RValue<'gcc> {
        let val_type = value.get_type();
        match (type_is_pointer(val_type), type_is_pointer(dest_ty)) {
            (false, true) => {
                // NOTE: Projecting a field of a pointer type will attemp a cast from a signed char to
                // a pointer, which is not supported by gccjit.
                return self.cx.context.new_cast(None, self.inttoptr(value, val_type.make_pointer()), dest_ty);
            },
            (false, false) => {
                // When they are not pointers, we want a transmute (or reinterpret_cast).
                self.bitcast(value, dest_ty)
            },
            (true, true) => self.cx.context.new_cast(None, value, dest_ty),
            (true, false) => unimplemented!(),
        }
    }

    /* Comparisons */
    fn icmp(&mut self, op: IntPredicate, mut lhs: RValue<'gcc>, mut rhs: RValue<'gcc>) -> RValue<'gcc> {
        let left_type = lhs.get_type();
        let right_type = rhs.get_type();
        if left_type != right_type {
            // NOTE: because libgccjit cannot compare function pointers.
            if left_type.dyncast_function_ptr_type().is_some() && right_type.dyncast_function_ptr_type().is_some() {
                lhs = self.context.new_cast(None, lhs, self.usize_type.make_pointer());
                rhs = self.context.new_cast(None, rhs, self.usize_type.make_pointer());
            }
            // NOTE: hack because we try to cast a vector type to the same vector type.
            else if format!("{:?}", left_type) != format!("{:?}", right_type) {
                rhs = self.context.new_cast(None, rhs, left_type);
            }
        }
        self.context.new_comparison(None, op.to_gcc_comparison(), lhs, rhs)
    }

    fn fcmp(&mut self, op: RealPredicate, lhs: RValue<'gcc>, rhs: RValue<'gcc>) -> RValue<'gcc> {
        self.context.new_comparison(None, op.to_gcc_comparison(), lhs, rhs)
    }

    /* Miscellaneous instructions */
    fn memcpy(&mut self, dst: RValue<'gcc>, dst_align: Align, src: RValue<'gcc>, src_align: Align, size: RValue<'gcc>, flags: MemFlags) {
        if flags.contains(MemFlags::NONTEMPORAL) {
            // HACK(nox): This is inefficient but there is no nontemporal memcpy.
            let val = self.load(src.get_type(), src, src_align);
            let ptr = self.pointercast(dst, self.type_ptr_to(self.val_ty(val)));
            self.store_with_flags(val, ptr, dst_align, flags);
            return;
        }
        let size = self.intcast(size, self.type_size_t(), false);
        let _is_volatile = flags.contains(MemFlags::VOLATILE);
        let dst = self.pointercast(dst, self.type_i8p());
        let src = self.pointercast(src, self.type_ptr_to(self.type_void()));
        let memcpy = self.context.get_builtin_function("memcpy");
        let block = self.block.expect("block");
        // TODO(antoyo): handle aligns and is_volatile.
        block.add_eval(None, self.context.new_call(None, memcpy, &[dst, src, size]));
    }

    fn memmove(&mut self, dst: RValue<'gcc>, dst_align: Align, src: RValue<'gcc>, src_align: Align, size: RValue<'gcc>, flags: MemFlags) {
        if flags.contains(MemFlags::NONTEMPORAL) {
            // HACK(nox): This is inefficient but there is no nontemporal memmove.
            let val = self.load(src.get_type(), src, src_align);
            let ptr = self.pointercast(dst, self.type_ptr_to(self.val_ty(val)));
            self.store_with_flags(val, ptr, dst_align, flags);
            return;
        }
        let size = self.intcast(size, self.type_size_t(), false);
        let _is_volatile = flags.contains(MemFlags::VOLATILE);
        let dst = self.pointercast(dst, self.type_i8p());
        let src = self.pointercast(src, self.type_ptr_to(self.type_void()));

        let memmove = self.context.get_builtin_function("memmove");
        let block = self.block.expect("block");
        // TODO(antoyo): handle is_volatile.
        block.add_eval(None, self.context.new_call(None, memmove, &[dst, src, size]));
    }

    fn memset(&mut self, ptr: RValue<'gcc>, fill_byte: RValue<'gcc>, size: RValue<'gcc>, _align: Align, flags: MemFlags) {
        let _is_volatile = flags.contains(MemFlags::VOLATILE);
        let ptr = self.pointercast(ptr, self.type_i8p());
        let memset = self.context.get_builtin_function("memset");
        let block = self.block.expect("block");
        // TODO(antoyo): handle align and is_volatile.
        let fill_byte = self.context.new_cast(None, fill_byte, self.i32_type);
        let size = self.intcast(size, self.type_size_t(), false);
        block.add_eval(None, self.context.new_call(None, memset, &[ptr, fill_byte, size]));
    }

    fn select(&mut self, cond: RValue<'gcc>, then_val: RValue<'gcc>, mut else_val: RValue<'gcc>) -> RValue<'gcc> {
        let func = self.current_func();
        let variable = func.new_local(None, then_val.get_type(), "selectVar");
        let then_block = func.new_block("then");
        let else_block = func.new_block("else");
        let after_block = func.new_block("after");
        self.llbb().end_with_conditional(None, cond, then_block, else_block);

        then_block.add_assignment(None, variable, then_val);
        then_block.end_with_jump(None, after_block);

        if then_val.get_type() != else_val.get_type() {
            else_val = self.context.new_cast(None, else_val, then_val.get_type());
        }
        else_block.add_assignment(None, variable, else_val);
        else_block.end_with_jump(None, after_block);

        // NOTE: since jumps were added in a place rustc does not expect, the current blocks in the
        // state need to be updated.
        self.block = Some(after_block);
        *self.cx.current_block.borrow_mut() = Some(after_block);

        variable.to_rvalue()
    }

    #[allow(dead_code)]
    fn va_arg(&mut self, _list: RValue<'gcc>, _ty: Type<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    fn extract_element(&mut self, _vec: RValue<'gcc>, _idx: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    fn vector_splat(&mut self, _num_elts: usize, _elt: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    fn extract_value(&mut self, aggregate_value: RValue<'gcc>, idx: u64) -> RValue<'gcc> {
        // FIXME(antoyo): it would be better if the API only called this on struct, not on arrays.
        assert_eq!(idx as usize as u64, idx);
        let value_type = aggregate_value.get_type();

        if value_type.dyncast_array().is_some() {
            let index = self.context.new_rvalue_from_long(self.u64_type, i64::try_from(idx).expect("i64::try_from"));
            let element = self.context.new_array_access(None, aggregate_value, index);
            element.get_address(None)
        }
        else if value_type.dyncast_vector().is_some() {
            panic!();
        }
        else if let Some(pointer_type) = value_type.get_pointee() {
            if let Some(struct_type) = pointer_type.is_struct() {
                // NOTE: hack to workaround a limitation of the rustc API: see comment on
                // CodegenCx.structs_as_pointer
                aggregate_value.dereference_field(None, struct_type.get_field(idx as i32)).to_rvalue()
            }
            else {
                panic!("Unexpected type {:?}", value_type);
            }
        }
        else if let Some(struct_type) = value_type.is_struct() {
            aggregate_value.access_field(None, struct_type.get_field(idx as i32)).to_rvalue()
        }
        else {
            panic!("Unexpected type {:?}", value_type);
        }
    }

    fn insert_value(&mut self, aggregate_value: RValue<'gcc>, value: RValue<'gcc>, idx: u64) -> RValue<'gcc> {
        // FIXME(antoyo): it would be better if the API only called this on struct, not on arrays.
        assert_eq!(idx as usize as u64, idx);
        let value_type = aggregate_value.get_type();

        let lvalue =
            if value_type.dyncast_array().is_some() {
                let index = self.context.new_rvalue_from_long(self.u64_type, i64::try_from(idx).expect("i64::try_from"));
                self.context.new_array_access(None, aggregate_value, index)
            }
            else if value_type.dyncast_vector().is_some() {
                panic!();
            }
            else if let Some(pointer_type) = value_type.get_pointee() {
                if let Some(struct_type) = pointer_type.is_struct() {
                    // NOTE: hack to workaround a limitation of the rustc API: see comment on
                    // CodegenCx.structs_as_pointer
                    aggregate_value.dereference_field(None, struct_type.get_field(idx as i32))
                }
                else {
                    panic!("Unexpected type {:?}", value_type);
                }
            }
            else {
                panic!("Unexpected type {:?}", value_type);
            };

        let lvalue_type = lvalue.to_rvalue().get_type();
        let value =
            // NOTE: sometimes, rustc will create a value with the wrong type.
            if lvalue_type != value.get_type() {
                self.context.new_cast(None, value, lvalue_type)
            }
            else {
                value
            };

        self.llbb().add_assignment(None, lvalue, value);

        aggregate_value
    }

    fn landing_pad(&mut self, _ty: Type<'gcc>, _pers_fn: RValue<'gcc>, _num_clauses: usize) -> RValue<'gcc> {
        let field1 = self.context.new_field(None, self.u8_type, "landing_pad_field_1");
        let field2 = self.context.new_field(None, self.i32_type, "landing_pad_field_1");
        let struct_type = self.context.new_struct_type(None, "landing_pad", &[field1, field2]);
        self.current_func().new_local(None, struct_type.as_type(), "landing_pad")
            .to_rvalue()
        // TODO(antoyo): Properly implement unwinding.
        // the above is just to make the compilation work as it seems
        // rustc_codegen_ssa now calls the unwinding builder methods even on panic=abort.
    }

    fn set_cleanup(&mut self, _landing_pad: RValue<'gcc>) {
        // TODO(antoyo)
    }

    fn resume(&mut self, _exn: RValue<'gcc>) -> RValue<'gcc> {
        unimplemented!();
    }

    fn cleanup_pad(&mut self, _parent: Option<RValue<'gcc>>, _args: &[RValue<'gcc>]) -> Funclet {
        unimplemented!();
    }

    fn cleanup_ret(&mut self, _funclet: &Funclet, _unwind: Option<Block<'gcc>>) -> RValue<'gcc> {
        unimplemented!();
    }

    fn catch_pad(&mut self, _parent: RValue<'gcc>, _args: &[RValue<'gcc>]) -> Funclet {
        unimplemented!();
    }

    fn catch_switch(&mut self, _parent: Option<RValue<'gcc>>, _unwind: Option<Block<'gcc>>, _num_handlers: usize) -> RValue<'gcc> {
        unimplemented!();
    }

    fn add_handler(&mut self, _catch_switch: RValue<'gcc>, _handler: Block<'gcc>) {
        unimplemented!();
    }

    fn set_personality_fn(&mut self, _personality: RValue<'gcc>) {
        // TODO(antoyo)
    }

    // Atomic Operations
    fn atomic_cmpxchg(&mut self, dst: RValue<'gcc>, cmp: RValue<'gcc>, src: RValue<'gcc>, order: AtomicOrdering, failure_order: AtomicOrdering, weak: bool) -> RValue<'gcc> {
        let expected = self.current_func().new_local(None, cmp.get_type(), "expected");
        self.llbb().add_assignment(None, expected, cmp);
        let success = self.compare_exchange(dst, expected, src, order, failure_order, weak);

        let pair_type = self.cx.type_struct(&[src.get_type(), self.bool_type], false);
        let result = self.current_func().new_local(None, pair_type, "atomic_cmpxchg_result");
        let align = Align::from_bits(64).expect("align"); // TODO(antoyo): use good align.

        let value_type = result.to_rvalue().get_type();
        if let Some(struct_type) = value_type.is_struct() {
            self.store(success, result.access_field(None, struct_type.get_field(1)).get_address(None), align);
            // NOTE: since success contains the call to the intrinsic, it must be stored before
            // expected so that we store expected after the call.
            self.store(expected.to_rvalue(), result.access_field(None, struct_type.get_field(0)).get_address(None), align);
        }
        // TODO(antoyo): handle when value is not a struct.

        result.to_rvalue()
    }

    fn atomic_rmw(&mut self, op: AtomicRmwBinOp, dst: RValue<'gcc>, src: RValue<'gcc>, order: AtomicOrdering) -> RValue<'gcc> {
        let size = self.cx.int_width(src.get_type()) / 8;
        let name =
            match op {
                AtomicRmwBinOp::AtomicXchg => format!("__atomic_exchange_{}", size),
                AtomicRmwBinOp::AtomicAdd => format!("__atomic_fetch_add_{}", size),
                AtomicRmwBinOp::AtomicSub => format!("__atomic_fetch_sub_{}", size),
                AtomicRmwBinOp::AtomicAnd => format!("__atomic_fetch_and_{}", size),
                AtomicRmwBinOp::AtomicNand => format!("__atomic_fetch_nand_{}", size),
                AtomicRmwBinOp::AtomicOr => format!("__atomic_fetch_or_{}", size),
                AtomicRmwBinOp::AtomicXor => format!("__atomic_fetch_xor_{}", size),
                AtomicRmwBinOp::AtomicMax => return self.atomic_extremum(ExtremumOperation::Max, dst, src, order),
                AtomicRmwBinOp::AtomicMin => return self.atomic_extremum(ExtremumOperation::Min, dst, src, order),
                AtomicRmwBinOp::AtomicUMax => return self.atomic_extremum(ExtremumOperation::Max, dst, src, order),
                AtomicRmwBinOp::AtomicUMin => return self.atomic_extremum(ExtremumOperation::Min, dst, src, order),
            };


        let atomic_function = self.context.get_builtin_function(name);
        let order = self.context.new_rvalue_from_int(self.i32_type, order.to_gcc());

        let void_ptr_type = self.context.new_type::<*mut ()>();
        let volatile_void_ptr_type = void_ptr_type.make_volatile();
        let dst = self.context.new_cast(None, dst, volatile_void_ptr_type);
        // FIXME(antoyo): not sure why, but we have the wrong type here.
        let new_src_type = atomic_function.get_param(1).to_rvalue().get_type();
        let src = self.context.new_cast(None, src, new_src_type);
        let res = self.context.new_call(None, atomic_function, &[dst, src, order]);
        self.context.new_cast(None, res, src.get_type())
    }

    fn atomic_fence(&mut self, order: AtomicOrdering, scope: SynchronizationScope) {
        let name =
            match scope {
                SynchronizationScope::SingleThread => "__atomic_signal_fence",
                SynchronizationScope::CrossThread => "__atomic_thread_fence",
            };
        let thread_fence = self.context.get_builtin_function(name);
        let order = self.context.new_rvalue_from_int(self.i32_type, order.to_gcc());
        self.llbb().add_eval(None, self.context.new_call(None, thread_fence, &[order]));
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

    fn call(&mut self, _typ: Type<'gcc>, func: RValue<'gcc>, args: &[RValue<'gcc>], funclet: Option<&Funclet>) -> RValue<'gcc> {
        // FIXME(antoyo): remove when having a proper API.
        let gcc_func = unsafe { std::mem::transmute(func) };
        if self.functions.borrow().values().find(|value| **value == gcc_func).is_some() {
            self.function_call(func, args, funclet)
        }
        else {
            // If it's a not function that was defined, it's a function pointer.
            self.function_ptr_call(func, args, funclet)
        }
    }

    fn zext(&mut self, value: RValue<'gcc>, dest_typ: Type<'gcc>) -> RValue<'gcc> {
        // FIXME(antoyo): this does not zero-extend.
        if value.get_type().is_bool() && dest_typ.is_i8(&self.cx) {
            // FIXME(antoyo): hack because base::from_immediate converts i1 to i8.
            // Fix the code in codegen_ssa::base::from_immediate.
            return value;
        }
        self.context.new_cast(None, value, dest_typ)
    }

    fn cx(&self) -> &CodegenCx<'gcc, 'tcx> {
        self.cx
    }

    fn apply_attrs_to_cleanup_callsite(&mut self, _llret: RValue<'gcc>) {
        unimplemented!();
    }

    fn set_span(&mut self, _span: Span) {}

    fn from_immediate(&mut self, val: Self::Value) -> Self::Value {
        if self.cx().val_ty(val) == self.cx().type_i1() {
            self.zext(val, self.cx().type_i8())
        }
        else {
            val
        }
    }

    fn to_immediate_scalar(&mut self, val: Self::Value, scalar: abi::Scalar) -> Self::Value {
        if scalar.is_bool() {
            return self.trunc(val, self.cx().type_i1());
        }
        val
    }

    fn fptoui_sat(&mut self, _val: RValue<'gcc>, _dest_ty: Type<'gcc>) -> Option<RValue<'gcc>> {
        None
    }

    fn fptosi_sat(&mut self, _val: RValue<'gcc>, _dest_ty: Type<'gcc>) -> Option<RValue<'gcc>> {
        None
    }

    fn instrprof_increment(&mut self, _fn_name: RValue<'gcc>, _hash: RValue<'gcc>, _num_counters: RValue<'gcc>, _index: RValue<'gcc>) {
        unimplemented!();
    }
}

impl<'a, 'gcc, 'tcx> Builder<'a, 'gcc, 'tcx> {
    pub fn shuffle_vector(&mut self, v1: RValue<'gcc>, v2: RValue<'gcc>, mask: RValue<'gcc>) -> RValue<'gcc> {
        let return_type = v1.get_type();
        let params = [
            self.context.new_parameter(None, return_type, "v1"),
            self.context.new_parameter(None, return_type, "v2"),
            self.context.new_parameter(None, mask.get_type(), "mask"),
        ];
        let shuffle = self.context.new_function(None, FunctionType::Extern, return_type, &params, "_mm_shuffle_epi8", false);
        self.context.new_call(None, shuffle, &[v1, v2, mask])
    }
}

impl<'a, 'gcc, 'tcx> StaticBuilderMethods for Builder<'a, 'gcc, 'tcx> {
    fn get_static(&mut self, def_id: DefId) -> RValue<'gcc> {
        // Forward to the `get_static` method of `CodegenCx`
        self.cx().get_static(def_id).get_address(None)
    }
}

impl<'tcx> HasParamEnv<'tcx> for Builder<'_, '_, 'tcx> {
    fn param_env(&self) -> ParamEnv<'tcx> {
        self.cx.param_env()
    }
}

impl<'tcx> HasTargetSpec for Builder<'_, '_, 'tcx> {
    fn target_spec(&self) -> &Target {
        &self.cx.target_spec()
    }
}

trait ToGccComp {
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

        let ordering =
            match self {
                AtomicOrdering::NotAtomic => __ATOMIC_RELAXED, // TODO(antoyo): check if that's the same.
                AtomicOrdering::Unordered => __ATOMIC_RELAXED,
                AtomicOrdering::Monotonic => __ATOMIC_RELAXED, // TODO(antoyo): check if that's the same.
                AtomicOrdering::Acquire => __ATOMIC_ACQUIRE,
                AtomicOrdering::Release => __ATOMIC_RELEASE,
                AtomicOrdering::AcquireRelease => __ATOMIC_ACQ_REL,
                AtomicOrdering::SequentiallyConsistent => __ATOMIC_SEQ_CST,
            };
        ordering as i32
    }
}
