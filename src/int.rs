//! Module to handle integer operations.
//! This module exists because some integer types are not supported on some gcc platforms, e.g.
//! 128-bit integers on 32-bit platforms and thus require to be handled manually.

use std::convert::TryFrom;

use gccjit::{ComparisonOp, FunctionType, RValue, ToRValue, Type, UnaryOp, BinaryOp};
use rustc_codegen_ssa::common::{IntPredicate, TypeKind};
use rustc_codegen_ssa::traits::{BackendTypes, BaseTypeMethods, BuilderMethods, OverflowOp};
use rustc_middle::ty::Ty;

use crate::builder::ToGccComp;
use crate::{builder::Builder, common::{SignType, TypeReflection}, context::CodegenCx};

impl<'a, 'gcc, 'tcx> Builder<'a, 'gcc, 'tcx> {
    pub fn gcc_urem(&self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // 128-bit unsigned %: __umodti3
        self.multiplicative_operation(BinaryOp::Modulo, "mod", false, a, b)
    }

    pub fn gcc_srem(&self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // 128-bit signed %:   __modti3
        self.multiplicative_operation(BinaryOp::Modulo, "mod", true, a, b)
    }

    pub fn gcc_not(&self, a: RValue<'gcc>) -> RValue<'gcc> {
        let typ = a.get_type();
        if self.is_native_int_type_or_bool(typ) {
            let operation =
                if typ.is_bool() {
                    UnaryOp::LogicalNegate
                }
                else {
                    UnaryOp::BitwiseNegate
                };
            self.cx.context.new_unary_op(None, operation, typ, a)
        }
        else {
            // TODO(antoyo): use __negdi2 and __negti2 instead?
            let element_type = typ.dyncast_array().expect("element type");
            let values = [
                self.cx.context.new_unary_op(None, UnaryOp::BitwiseNegate, element_type, self.low(a)),
                self.cx.context.new_unary_op(None, UnaryOp::BitwiseNegate, element_type, self.high(a)),
            ];
            self.cx.context.new_array_constructor(None, typ, &values)
        }
    }

    pub fn gcc_neg(&self, a: RValue<'gcc>) -> RValue<'gcc> {
        let a_type = a.get_type();
        if self.is_native_int_type(a_type) {
            self.cx.context.new_unary_op(None, UnaryOp::Minus, a.get_type(), a)
        }
        else {
            let param_a = self.context.new_parameter(None, a_type, "a");
            let func = self.context.new_function(None, FunctionType::Extern, a_type, &[param_a], "__negti2", false);
            self.context.new_call(None, func, &[a])
        }
    }

    pub fn gcc_and(&self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.cx.bitwise_operation(BinaryOp::BitwiseAnd, a, b)
    }

    pub fn gcc_lshr(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        let a_type = a.get_type();
        let b_type = b.get_type();
        let a_native = self.is_native_int_type(a_type);
        let b_native = self.is_native_int_type(b_type);
        if a_native && b_native {
            // FIXME(antoyo): remove the casts when libgccjit can shift an unsigned number by a signed number.
            // TODO(antoyo): cast to unsigned to do a logical shift if that does not work.
            if a_type.is_signed(self) != b_type.is_signed(self) {
                let b = self.context.new_cast(None, b, a_type);
                a >> b
            }
            else {
                a >> b
            }
        }
        else if a_native && !b_native {
            self.gcc_lshr(a, self.gcc_int_cast(b, a_type))
        }
        else {
            // NOTE: we cannot use the lshr builtin because it's calling hi() (to get the most
            // significant half of the number) which uses lshr.

            let native_int_type = a_type.dyncast_array().expect("get element type");

            let func = self.current_func();
            let then_block = func.new_block("then");
            let else_block = func.new_block("else");
            let after_block = func.new_block("after");
            let b0_block = func.new_block("b0");
            let actual_else_block = func.new_block("actual_else");

            let result = func.new_local(None, a_type, "shiftResult");

            let sixty_four = self.gcc_int(native_int_type, 64);
            let sixty_three = self.gcc_int(native_int_type, 63);
            let zero = self.gcc_zero(native_int_type);
            let b = self.gcc_int_cast(b, native_int_type);
            let condition = self.gcc_icmp(IntPredicate::IntNE, self.gcc_and(b, sixty_four), zero);
            self.llbb().end_with_conditional(None, condition, then_block, else_block);

            // TODO(antoyo): take endianness into account.
            let shift_value = self.gcc_sub(b, sixty_four);
            let high = self.high(a);
            let sign =
                if a_type.is_signed(self) {
                    high >> sixty_three
                }
                else {
                    zero
                };
            let values = [
                high >> shift_value,
                sign,
            ];
            let array_value = self.context.new_array_constructor(None, a_type, &values);
            then_block.add_assignment(None, result, array_value);
            then_block.end_with_jump(None, after_block);

            let condition = self.gcc_icmp(IntPredicate::IntEQ, b, zero);
            else_block.end_with_conditional(None, condition, b0_block, actual_else_block);

            b0_block.add_assignment(None, result, a);
            b0_block.end_with_jump(None, after_block);

            let shift_value = self.gcc_sub(sixty_four, b);
            // NOTE: cast low to its unsigned type in order to perform a logical right shift.
            let unsigned_type = native_int_type.to_unsigned(&self.cx);
            let casted_low = self.context.new_cast(None, self.low(a), unsigned_type);
            let shifted_low = casted_low >> self.context.new_cast(None, b, unsigned_type);
            let shifted_low = self.context.new_cast(None, shifted_low, native_int_type);
            let values = [
                (high << shift_value) | shifted_low,
                high >> b,
            ];
            let array_value = self.context.new_array_constructor(None, a_type, &values);
            actual_else_block.add_assignment(None, result, array_value);
            actual_else_block.end_with_jump(None, after_block);

            // NOTE: since jumps were added in a place rustc does not expect, the current block in the
            // state need to be updated.
            self.switch_to_block(after_block);

            result.to_rvalue()
        }
    }

    fn additive_operation(&self, operation: BinaryOp, a: RValue<'gcc>, mut b: RValue<'gcc>) -> RValue<'gcc> {
        let a_type = a.get_type();
        let b_type = b.get_type();
        if self.is_native_int_type_or_bool(a_type) && self.is_native_int_type_or_bool(b_type) {
            if a_type != b_type {
                if a_type.is_vector() {
                    // Vector types need to be bitcast.
                    // TODO(antoyo): perhaps use __builtin_convertvector for vector casting.
                    b = self.context.new_bitcast(None, b, a.get_type());
                }
                else {
                    b = self.context.new_cast(None, b, a.get_type());
                }
            }
            self.context.new_binary_op(None, operation, a_type, a, b)
        }
        else {
            let signed = a_type.is_compatible_with(self.i128_type);
            let func_name =
                match (operation, signed) {
                    (BinaryOp::Plus, true) => "__rust_i128_add",
                    (BinaryOp::Plus, false) => "__rust_u128_add",
                    (BinaryOp::Minus, true) => "__rust_i128_sub",
                    (BinaryOp::Minus, false) => "__rust_u128_sub",
                    _ => unreachable!("unexpected additive operation {:?}", operation),
                };
            let param_a = self.context.new_parameter(None, a_type, "a");
            let param_b = self.context.new_parameter(None, b_type, "b");
            let func = self.context.new_function(None, FunctionType::Extern, a_type, &[param_a, param_b], func_name, false);
            self.context.new_call(None, func, &[a, b])
        }
    }

    pub fn gcc_add(&self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.additive_operation(BinaryOp::Plus, a, b)
    }

    pub fn gcc_mul(&self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.multiplicative_operation(BinaryOp::Mult, "mul", true, a, b)
    }

    pub fn gcc_sub(&self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.additive_operation(BinaryOp::Minus, a, b)
    }

    fn multiplicative_operation(&self, operation: BinaryOp, operation_name: &str, signed: bool, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        let a_type = a.get_type();
        let b_type = b.get_type();
        if self.is_native_int_type_or_bool(a_type) && self.is_native_int_type_or_bool(b_type) {
            self.context.new_binary_op(None, operation, a_type, a, b)
        }
        else {
            let sign =
                if signed {
                    ""
                }
                else {
                    "u"
                };
            let func_name = format!("__{}{}ti3", sign, operation_name);
            let param_a = self.context.new_parameter(None, a_type, "a");
            let param_b = self.context.new_parameter(None, b_type, "b");
            let func = self.context.new_function(None, FunctionType::Extern, a_type, &[param_a, param_b], func_name, false);
            self.context.new_call(None, func, &[a, b])
        }
    }

    pub fn gcc_sdiv(&self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // TODO(antoyo): check if the types are signed?
        // 128-bit, signed: __divti3
        // TODO(antoyo): convert the arguments to signed?
        self.multiplicative_operation(BinaryOp::Divide, "div", true, a, b)
    }

    pub fn gcc_udiv(&self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        // 128-bit, unsigned: __udivti3
        self.multiplicative_operation(BinaryOp::Divide, "div", false, a, b)
    }

    pub fn gcc_checked_binop(&self, oop: OverflowOp, typ: Ty<'_>, lhs: <Self as BackendTypes>::Value, rhs: <Self as BackendTypes>::Value) -> (<Self as BackendTypes>::Value, <Self as BackendTypes>::Value) {
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
            if self.is_native_int_type(lhs.get_type()) {
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
                }
            }
            else {
                match new_kind {
                    Int(I128) | Uint(U128) => {
                        let func_name =
                            match oop {
                                OverflowOp::Add =>
                                    match new_kind {
                                        Int(I128) => "__rust_i128_addo",
                                        Uint(U128) => "__rust_u128_addo",
                                        _ => unreachable!(),
                                    },
                                OverflowOp::Sub =>
                                    match new_kind {
                                        Int(I128) => "__rust_i128_subo",
                                        Uint(U128) => "__rust_u128_subo",
                                        _ => unreachable!(),
                                    },
                                OverflowOp::Mul =>
                                    match new_kind {
                                        Int(I128) => "__rust_i128_mulo", // TODO(antoyo): use __muloti4d instead?
                                        Uint(U128) => "__rust_u128_mulo",
                                        _ => unreachable!(),
                                    },
                            };
                        let a_type = lhs.get_type();
                        let b_type = rhs.get_type();
                        let param_a = self.context.new_parameter(None, a_type, "a");
                        let param_b = self.context.new_parameter(None, b_type, "b");
                        let result_field = self.context.new_field(None, a_type, "result");
                        let overflow_field = self.context.new_field(None, self.bool_type, "overflow");
                        let return_type = self.context.new_struct_type(None, "result_overflow", &[result_field, overflow_field]);
                        let func = self.context.new_function(None, FunctionType::Extern, return_type.as_type(), &[param_a, param_b], func_name, false);
                        let result = self.context.new_call(None, func, &[lhs, rhs]);
                        let overflow = result.access_field(None, overflow_field);
                        let int_result = result.access_field(None, result_field);
                        return (int_result, overflow);
                    },
                    _ => {
                        match oop {
                            OverflowOp::Mul =>
                                match new_kind {
                                    Int(I32) => "__mulosi4",
                                    Int(I64) => "__mulodi4",
                                    _ => unreachable!(),
                                },
                            _ => unimplemented!("overflow operation for {:?}", new_kind),
                        }
                    }
                }
            };

        let intrinsic = self.context.get_builtin_function(&name);
        let res = self.current_func()
            // TODO(antoyo): is it correct to use rhs type instead of the parameter typ?
            .new_local(None, rhs.get_type(), "binopResult")
            .get_address(None);
        let overflow = self.overflow_call(intrinsic, &[lhs, rhs, res], None);
        (res.dereference(None).to_rvalue(), overflow)
    }

    pub fn gcc_icmp(&self, op: IntPredicate, mut lhs: RValue<'gcc>, mut rhs: RValue<'gcc>) -> RValue<'gcc> {
        let a_type = lhs.get_type();
        let b_type = rhs.get_type();
        if self.is_non_native_int_type(a_type) || self.is_non_native_int_type(b_type) {
            let signed = a_type.is_compatible_with(self.i128_type);
            let sign =
                if signed {
                    ""
                }
                else {
                    "u"
                };
            let func_name = format!("__{}cmpti2", sign);
            let param_a = self.context.new_parameter(None, a_type, "a");
            let param_b = self.context.new_parameter(None, b_type, "b");
            let func = self.context.new_function(None, FunctionType::Extern, self.int_type, &[param_a, param_b], func_name, false);
            let cmp = self.context.new_call(None, func, &[lhs, rhs]);
            let (op, limit) =
                match op {
                    IntPredicate::IntEQ => {
                        return self.context.new_comparison(None, ComparisonOp::Equals, cmp, self.context.new_rvalue_one(self.int_type));
                    },
                    IntPredicate::IntNE => {
                        return self.context.new_comparison(None, ComparisonOp::NotEquals, cmp, self.context.new_rvalue_one(self.int_type));
                    },
                    IntPredicate::IntUGT => (ComparisonOp::Equals, 2),
                    IntPredicate::IntUGE => (ComparisonOp::GreaterThanEquals, 1),
                    IntPredicate::IntULT => (ComparisonOp::Equals, 0),
                    IntPredicate::IntULE => (ComparisonOp::LessThanEquals, 1),
                    IntPredicate::IntSGT => (ComparisonOp::Equals, 2),
                    IntPredicate::IntSGE => (ComparisonOp::GreaterThanEquals, 1),
                    IntPredicate::IntSLT => (ComparisonOp::Equals, 0),
                    IntPredicate::IntSLE => (ComparisonOp::LessThanEquals, 1),
                };
            self.context.new_comparison(None, op, cmp, self.context.new_rvalue_from_int(self.int_type, limit))
        }
        else if a_type.get_pointee().is_some() && b_type.get_pointee().is_some() {
            // NOTE: gcc cannot compare pointers to different objects, but rustc does that, so cast them to usize.
            lhs = self.context.new_bitcast(None, lhs, self.usize_type);
            rhs = self.context.new_bitcast(None, rhs, self.usize_type);
            self.context.new_comparison(None, op.to_gcc_comparison(), lhs, rhs)
        }
        else {
            if a_type != b_type {
                // NOTE: because libgccjit cannot compare function pointers.
                if a_type.dyncast_function_ptr_type().is_some() && b_type.dyncast_function_ptr_type().is_some() {
                    lhs = self.context.new_cast(None, lhs, self.usize_type.make_pointer());
                    rhs = self.context.new_cast(None, rhs, self.usize_type.make_pointer());
                }
                // NOTE: hack because we try to cast a vector type to the same vector type.
                else if format!("{:?}", a_type) != format!("{:?}", b_type) {
                    rhs = self.context.new_cast(None, rhs, a_type);
                }
            }
            self.context.new_comparison(None, op.to_gcc_comparison(), lhs, rhs)
        }
    }

    pub fn gcc_xor(&self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        let a_type = a.get_type();
        let b_type = b.get_type();
        if self.is_native_int_type_or_bool(a_type) && self.is_native_int_type_or_bool(b_type) {
            a ^ b
        }
        else {
            let values = [
                self.low(a) ^ self.low(b),
                self.high(a) ^ self.high(b),
            ];
            self.context.new_array_constructor(None, a_type, &values)
        }
    }

    pub fn gcc_shl(&mut self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        let a_type = a.get_type();
        let b_type = b.get_type();
        let a_native = self.is_native_int_type(a_type);
        let b_native = self.is_native_int_type(b_type);
        if a_native && b_native {
            // FIXME(antoyo): remove the casts when libgccjit can shift an unsigned number by an unsigned number.
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
        else if a_native && !b_native {
            self.gcc_shl(a, self.gcc_int_cast(b, a_type))
        }
        else {
            // NOTE: we cannot use the ashl builtin because it's calling widen_hi() which uses ashl.
            let native_int_type = a_type.dyncast_array().expect("get element type");

            let func = self.current_func();
            let then_block = func.new_block("then");
            let else_block = func.new_block("else");
            let after_block = func.new_block("after");
            let b0_block = func.new_block("b0");
            let actual_else_block = func.new_block("actual_else");

            let result = func.new_local(None, a_type, "shiftResult");

            let b = self.gcc_int_cast(b, native_int_type);
            let sixty_four = self.gcc_int(native_int_type, 64);
            let zero = self.gcc_zero(native_int_type);
            let condition = self.gcc_icmp(IntPredicate::IntNE, self.gcc_and(b, sixty_four), zero);
            self.llbb().end_with_conditional(None, condition, then_block, else_block);

            // TODO(antoyo): take endianness into account.
            let values = [
                zero,
                self.low(a) << (b - sixty_four),
            ];
            let array_value = self.context.new_array_constructor(None, a_type, &values);
            then_block.add_assignment(None, result, array_value);
            then_block.end_with_jump(None, after_block);

            let condition = self.gcc_icmp(IntPredicate::IntEQ, b, zero);
            else_block.end_with_conditional(None, condition, b0_block, actual_else_block);

            b0_block.add_assignment(None, result, a);
            b0_block.end_with_jump(None, after_block);

            // NOTE: cast low to its unsigned type in order to perform a logical right shift.
            let unsigned_type = native_int_type.to_unsigned(&self.cx);
            let casted_low = self.context.new_cast(None, self.low(a), unsigned_type);
            let shift_value = self.context.new_cast(None, sixty_four - b, unsigned_type);
            let high_low = self.context.new_cast(None, casted_low >> shift_value, native_int_type);
            let values = [
                self.low(a) << b,
                (self.high(a) << b) | high_low,
            ];

            let array_value = self.context.new_array_constructor(None, a_type, &values);
            actual_else_block.add_assignment(None, result, array_value);
            actual_else_block.end_with_jump(None, after_block);

            // NOTE: since jumps were added in a place rustc does not expect, the current block in the
            // state need to be updated.
            self.switch_to_block(after_block);

            result.to_rvalue()
        }
    }

    pub fn gcc_bswap(&mut self, mut arg: RValue<'gcc>, width: u64) -> RValue<'gcc> {
        let arg_type = arg.get_type();
        if !self.is_native_int_type(arg_type) {
            let native_int_type = arg_type.dyncast_array().expect("get element type");
            let lsb = self.context.new_array_access(None, arg, self.context.new_rvalue_from_int(self.int_type, 0)).to_rvalue();
            let swapped_lsb = self.gcc_bswap(lsb, width / 2);
            let swapped_lsb = self.context.new_cast(None, swapped_lsb, native_int_type);
            let msb = self.context.new_array_access(None, arg, self.context.new_rvalue_from_int(self.int_type, 1)).to_rvalue();
            let swapped_msb = self.gcc_bswap(msb, width / 2);
            let swapped_msb = self.context.new_cast(None, swapped_msb, native_int_type);

            // NOTE: we also need to swap the two elements here, in addition to swapping inside
            // the elements themselves like done above.
            return self.context.new_array_constructor(None, arg_type, &[swapped_msb, swapped_lsb]);
        }

        // TODO(antoyo): check if it's faster to use string literals and a
        // match instead of format!.
        let bswap = self.cx.context.get_builtin_function(&format!("__builtin_bswap{}", width));
        // FIXME(antoyo): this cast should not be necessary. Remove
        // when having proper sized integer types.
        let param_type = bswap.get_param(0).to_rvalue().get_type();
        if param_type != arg_type {
            arg = self.bitcast(arg, param_type);
        }
        self.cx.context.new_call(None, bswap, &[arg])
    }
}

impl<'gcc, 'tcx> CodegenCx<'gcc, 'tcx> {
    pub fn gcc_int(&self, typ: Type<'gcc>, int: i64) -> RValue<'gcc> {
        if self.is_native_int_type_or_bool(typ) {
            self.context.new_rvalue_from_long(typ, i64::try_from(int).expect("i64::try_from"))
        }
        else {
            // NOTE: set the sign in high.
            self.from_low_high(typ, int, -(int.is_negative() as i64))
        }
    }

    pub fn gcc_uint(&self, typ: Type<'gcc>, int: u64) -> RValue<'gcc> {
        if self.is_native_int_type_or_bool(typ) {
            self.context.new_rvalue_from_long(typ, u64::try_from(int).expect("u64::try_from") as i64)
        }
        else {
            self.from_low_high(typ, int as i64, 0)
        }
    }

    pub fn gcc_uint_big(&self, typ: Type<'gcc>, num: u128) -> RValue<'gcc> {
        let low = num as u64;
        let high = (num >> 64) as u64;
        if num >> 64 != 0 {
            // FIXME(antoyo): use a new function new_rvalue_from_unsigned_long()?
            if self.is_native_int_type(typ) {
                let low = self.context.new_rvalue_from_long(self.u64_type, low as i64);
                let high = self.context.new_rvalue_from_long(typ, high as i64);

                let sixty_four = self.context.new_rvalue_from_long(typ, 64);
                let shift = high << sixty_four;
                shift | self.context.new_cast(None, low, typ)
            }
            else {
                self.from_low_high(typ, low as i64, high as i64)
            }
        }
        else if typ.is_i128(self) {
            let num = self.context.new_rvalue_from_long(self.u64_type, num as u64 as i64);
            self.gcc_int_cast(num, typ)
        }
        else {
            self.gcc_uint(typ, num as u64)
        }
    }

    pub fn gcc_zero(&self, typ: Type<'gcc>) -> RValue<'gcc> {
        if self.is_native_int_type_or_bool(typ) {
            self.context.new_rvalue_zero(typ)
        }
        else {
            self.from_low_high(typ, 0, 0)
        }
    }

    pub fn gcc_int_width(&self, typ: Type<'gcc>) -> u64 {
        if self.is_native_int_type_or_bool(typ) {
            typ.get_size() as u64 * 8
        }
        else {
            // NOTE: the only unsupported types are u128 and i128.
            128
        }
    }

    fn bitwise_operation(&self, operation: BinaryOp, a: RValue<'gcc>, mut b: RValue<'gcc>) -> RValue<'gcc> {
        let a_type = a.get_type();
        let b_type = b.get_type();
        let a_native = self.is_native_int_type_or_bool(a_type);
        let b_native = self.is_native_int_type_or_bool(b_type);
        if a_type.is_vector() && b_type.is_vector() {
            self.context.new_binary_op(None, operation, a_type, a, b)
        }
        else if a_native && b_native {
            if a_type != b_type {
                b = self.context.new_cast(None, b, a_type);
            }
            self.context.new_binary_op(None, operation, a_type, a, b)
        }
        else {
            assert!(!a_native && !b_native, "both types should either be native or non-native for or operation");
            let native_int_type = a_type.dyncast_array().expect("get element type");
            let values = [
                self.context.new_binary_op(None, operation, native_int_type, self.low(a), self.low(b)),
                self.context.new_binary_op(None, operation, native_int_type, self.high(a), self.high(b)),
            ];
            self.context.new_array_constructor(None, a_type, &values)
        }
    }

    pub fn gcc_or(&self, a: RValue<'gcc>, b: RValue<'gcc>) -> RValue<'gcc> {
        self.bitwise_operation(BinaryOp::BitwiseOr, a, b)
    }

    // TODO(antoyo): can we use https://github.com/rust-lang/compiler-builtins/blob/master/src/int/mod.rs#L379 instead?
    pub fn gcc_int_cast(&self, value: RValue<'gcc>, dest_typ: Type<'gcc>) -> RValue<'gcc> {
        let value_type = value.get_type();
        if self.is_native_int_type_or_bool(dest_typ) && self.is_native_int_type_or_bool(value_type) {
            self.context.new_cast(None, value, dest_typ)
        }
        else if self.is_native_int_type_or_bool(dest_typ) {
            self.context.new_cast(None, self.low(value), dest_typ)
        }
        else if self.is_native_int_type_or_bool(value_type) {
            let dest_element_type = dest_typ.dyncast_array().expect("get element type");

            // NOTE: set the sign of the value.
            let zero = self.context.new_rvalue_zero(value_type);
            let is_negative = self.context.new_comparison(None, ComparisonOp::LessThan, value, zero);
            let is_negative = self.gcc_int_cast(is_negative, dest_element_type);
            let values = [
                self.context.new_cast(None, value, dest_element_type),
                self.context.new_unary_op(None, UnaryOp::Minus, dest_element_type, is_negative),
            ];
            self.context.new_array_constructor(None, dest_typ, &values)
        }
        else {
            // Since u128 and i128 are the only types that can be unsupported, we know the type of
            // value and the destination type have the same size, so a bitcast is fine.

            // TODO(antoyo): perhaps use __builtin_convertvector for vector casting.
            self.context.new_bitcast(None, value, dest_typ)
        }
    }

    fn int_to_float_cast(&self, signed: bool, value: RValue<'gcc>, dest_typ: Type<'gcc>) -> RValue<'gcc> {
        let value_type = value.get_type();
        if self.is_native_int_type_or_bool(value_type) {
            return self.context.new_cast(None, value, dest_typ);
        }

        let name_suffix =
            match self.type_kind(dest_typ) {
                TypeKind::Float => "tisf",
                TypeKind::Double => "tidf",
                kind => panic!("cannot cast a non-native integer to type {:?}", kind),
            };
        let sign =
            if signed {
                ""
            }
            else {
                "un"
            };
        let func_name = format!("__float{}{}", sign, name_suffix);
        let param = self.context.new_parameter(None, value_type, "n");
        let func = self.context.new_function(None, FunctionType::Extern, dest_typ, &[param], func_name, false);
        self.context.new_call(None, func, &[value])
    }

    pub fn gcc_int_to_float_cast(&self, value: RValue<'gcc>, dest_typ: Type<'gcc>) -> RValue<'gcc> {
        self.int_to_float_cast(true, value, dest_typ)
    }

    pub fn gcc_uint_to_float_cast(&self, value: RValue<'gcc>, dest_typ: Type<'gcc>) -> RValue<'gcc> {
        self.int_to_float_cast(false, value, dest_typ)
    }

    fn float_to_int_cast(&self, signed: bool, value: RValue<'gcc>, dest_typ: Type<'gcc>) -> RValue<'gcc> {
        let value_type = value.get_type();
        if self.is_native_int_type_or_bool(dest_typ) {
            return self.context.new_cast(None, value, dest_typ);
        }

        let name_suffix =
            match self.type_kind(value_type) {
                TypeKind::Float => "sfti",
                TypeKind::Double => "dfti",
                kind => panic!("cannot cast a {:?} to non-native integer", kind),
            };
        let sign =
            if signed {
                ""
            }
            else {
                "uns"
            };
        let func_name = format!("__fix{}{}", sign, name_suffix);
        let param = self.context.new_parameter(None, value_type, "n");
        let func = self.context.new_function(None, FunctionType::Extern, dest_typ, &[param], func_name, false);
        self.context.new_call(None, func, &[value])
    }

    pub fn gcc_float_to_int_cast(&self, value: RValue<'gcc>, dest_typ: Type<'gcc>) -> RValue<'gcc> {
        self.float_to_int_cast(true, value, dest_typ)
    }

    pub fn gcc_float_to_uint_cast(&self, value: RValue<'gcc>, dest_typ: Type<'gcc>) -> RValue<'gcc> {
        self.float_to_int_cast(false, value, dest_typ)
    }

    fn high(&self, value: RValue<'gcc>) -> RValue<'gcc> {
        self.context.new_array_access(None, value, self.context.new_rvalue_from_int(self.int_type, 1))
            .to_rvalue()
    }

    fn low(&self, value: RValue<'gcc>) -> RValue<'gcc> {
        self.context.new_array_access(None, value, self.context.new_rvalue_from_int(self.int_type, 0))
            .to_rvalue()
    }

    fn from_low_high(&self, typ: Type<'gcc>, low: i64, high: i64) -> RValue<'gcc> {
        let native_int_type = typ.dyncast_array().expect("get element type");
        let values = [
            self.context.new_rvalue_from_long(native_int_type, low),
            self.context.new_rvalue_from_long(native_int_type, high),
        ];
        self.context.new_array_constructor(None, typ, &values)
    }
}
