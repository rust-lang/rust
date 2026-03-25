/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use melior::Context;
use melior::dialect::ods::arith::{
    AddFOperation, AddIOperation, CmpIOperation, ConstantOperation, ExtSIOperation, MulIOperation,
};
use melior::ir::attribute::IntegerAttribute;
use melior::ir::r#type::{IntegerType, RankedTensorType};
use melior::ir::{Attribute, Location, Type, TypeLike, Value, ValueLike};
use rustc_ast::{IntTy, UintTy};
use rustc_middle::ty::{ScalarInt, Ty, TyKind};

use crate::errors::Error;

pub struct Predicate(i32);
impl Predicate {
    pub const EQ: Predicate = Self(0);
    pub const NE: Predicate = Self(1);
    pub const SLT: Predicate = Self(2);
    pub const SLE: Predicate = Self(3);
    pub const SGT: Predicate = Self(4);
    pub const SGE: Predicate = Self(5);
    pub const ULT: Predicate = Self(6);
    pub const ULE: Predicate = Self(7);
    pub const UGT: Predicate = Self(8);
    pub const UGE: Predicate = Self(9);
}

impl std::fmt::Display for Predicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// This enum represents integers but we use unsigned integers for the values,
/// as we use the signless variant of the integer type in MLIR.
pub enum Int {
    I8(u8),
    I16(u16),
    I32(u32),
    I64(u64),
    I128(u128),
    Isize(u64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    Usize(u64),
}

impl Int {
    pub fn from_scalar<'tcx>(ty: Ty<'tcx>, scalar: ScalarInt) -> Result<Self, Error> {
        let value = match ty.kind() {
            TyKind::Int(int_ty) => match int_ty {
                IntTy::I8 => Int::I8(scalar.to_u8()),
                IntTy::I16 => Int::I16(scalar.to_u16()),
                IntTy::I32 => Int::I32(scalar.to_u32()),
                IntTy::I64 => Int::I64(scalar.to_u64()),
                IntTy::I128 => Int::I128(scalar.to_u128()),
                IntTy::Isize => Int::Isize(scalar.to_u64()),
            },
            TyKind::Uint(uint_ty) => match uint_ty {
                UintTy::U8 => Int::U8(scalar.to_u8()),
                UintTy::U16 => Int::U16(scalar.to_u16()),
                UintTy::U32 => Int::U32(scalar.to_u32()),
                UintTy::U64 => Int::U64(scalar.to_u64()),
                UintTy::U128 => Int::U128(scalar.to_u128()),
                UintTy::Usize => Int::Usize(scalar.to_u64()),
            },
            _ => {
                return Err(Error::InvalidType {
                    msg: format!("Unsupported type for constant: {:?}", ty),
                });
            }
        };

        Ok(value)
    }

    pub fn ty<'ctx>(&self, context: &'ctx Context) -> IntegerType<'ctx> {
        let num_bits = match self {
            Int::I8(_) => 8,
            Int::I16(_) => 16,
            Int::I32(_) => 32,
            Int::I64(_) => 64,
            Int::I128(_) => 128,
            Int::U8(_) => 8,
            Int::U16(_) => 16,
            Int::U32(_) => 32,
            Int::U64(_) => 64,
            Int::U128(_) => 128,
            Int::Isize(_) => 64, // isize is treated as i64
            Int::Usize(_) => 64, // usize is treated as i64
        };

        IntegerType::new(context, num_bits)
    }

    pub fn attr<'ctx>(&self, context: &'ctx Context) -> Attribute<'ctx> {
        // NOTE: unsigned values are promotied of i64
        let source_attr = match self {
            Int::I8(value) => format!("{} : i8", value),
            Int::I16(value) => format!("{} : i16", value),
            Int::I32(value) => format!("{} : i32", value),
            Int::I64(value) => format!("{} : i64", value),
            Int::I128(value) => format!("{} : i128", value),
            Int::Isize(value) => format!("{} : i64", value),
            Int::U8(value) => format!("{} : i8", value),
            Int::U16(value) => format!("{} : i16", value),
            Int::U32(value) => format!("{} : i32", value),
            Int::U64(value) => format!("{} : i64", value),
            Int::U128(value) => format!("{} : i128", value),
            Int::Usize(value) => format!("{} : i64", value),
        };

        Attribute::parse(context, &source_attr).unwrap()
    }
}

pub fn create_int_constant<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    value: Int,
) -> Result<ConstantOperation<'ctx>, Error> {
    let ty = value.ty(context).into();
    let num_attr = value.attr(context);

    create_constant(context, location, num_attr, ty)
}

pub fn create_constant<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    attr: Attribute<'ctx>,
    result_ty: Type<'ctx>,
) -> Result<ConstantOperation<'ctx>, Error> {
    Ok(ConstantOperation::builder(context, location).value(attr).result(result_ty).build())
}

pub fn create_muli<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    lhs: Value<'ctx, 'ctx>,
    rhs: Value<'ctx, 'ctx>,
) -> Result<MulIOperation<'ctx>, Error> {
    let lhs_ty = lhs.r#type();
    let rhs_ty = rhs.r#type();

    if lhs_ty != rhs_ty {
        return Err(Error::IncompatibleTypes { lhs: lhs_ty.to_string(), rhs: rhs_ty.to_string() });
    }

    if !lhs_ty.is_integer() {
        return Err(Error::InvalidType { msg: lhs_ty.to_string() });
    }

    Ok(MulIOperation::builder(context, location).lhs(lhs).rhs(rhs).build())
}

pub fn create_addi<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    lhs: Value<'ctx, 'ctx>,
    rhs: Value<'ctx, 'ctx>,
) -> Result<AddIOperation<'ctx>, Error> {
    Ok(AddIOperation::builder(context, location).lhs(lhs).rhs(rhs).build())
}

pub fn create_addf<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    lhs: Value<'ctx, 'ctx>,
    rhs: Value<'ctx, 'ctx>,
) -> Result<AddFOperation<'ctx>, Error> {
    Ok(AddFOperation::builder(context, location).lhs(lhs).rhs(rhs).build())
}

pub fn create_extsi<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, 'ctx>,
    result_ty: Type<'ctx>,
) -> Result<ExtSIOperation<'ctx>, Error> {
    Ok(ExtSIOperation::builder(context, location).r#in(src).out(result_ty).build())
}

pub fn create_cmpi<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    predicate: Predicate,
    lhs: Value<'ctx, 'ctx>,
    rhs: Value<'ctx, 'ctx>,
    result_ty: Type<'ctx>,
) -> Result<CmpIOperation<'ctx>, Error> {
    let predicate = Attribute::parse(context, &predicate.to_string()).unwrap();

    Ok(CmpIOperation::builder(context, location)
        .predicate(predicate)
        .lhs(lhs)
        .rhs(rhs)
        .result(result_ty)
        .build())
}
#[cfg(test)]
mod tests {

    use melior::ir::operation::OperationLike;
    use melior::ir::{BlockLike, Location, Module, Operation};
    use rstest::rstest;

    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn test_create_constant() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        let constant_op = create_int_constant(&context, location, Int::I64(253)).unwrap();

        let expected = "%c253_i64 = arith.constant 253 : i64\n";
        let output = constant_op.as_operation().to_string();
        assert_eq!(expected, output);
    }

    #[test]
    fn test_create_muli() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        // Create two i32 constants
        let lhs: Operation = create_int_constant(&context, location, Int::I32(4)).unwrap().into();
        let rhs: Operation = create_int_constant(&context, location, Int::I32(5)).unwrap().into();

        // Get their values
        let lhs_value = lhs.result(0).unwrap().into();
        let rhs_value = rhs.result(0).unwrap().into();

        // Generate arith.muli operation
        let muli = create_muli(&context, location, lhs_value, rhs_value).unwrap().into();

        module.body().append_operation(lhs);
        module.body().append_operation(rhs);
        module.body().append_operation(muli);

        let expected = "module {\n  %c4_i32 = arith.constant 4 : i32\n  %c5_i32 = arith.constant 5 : i32\n  %0 = arith.muli %c4_i32, %c5_i32 : i32\n}\n";
        let output = module.as_operation().to_string();
        assert_eq!(expected, output);
    }

    #[test]
    fn test_create_addi() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        // Create two i32 constants
        let lhs: Operation = create_int_constant(&context, location, Int::I32(4)).unwrap().into();
        let rhs: Operation = create_int_constant(&context, location, Int::I32(5)).unwrap().into();

        // Get their values
        let lhs_value = lhs.result(0).unwrap().into();
        let rhs_value = rhs.result(0).unwrap().into();

        // Generate arith.muli operation
        let addi = create_addi(&context, location, lhs_value, rhs_value).unwrap().into();

        module.body().append_operation(lhs);
        module.body().append_operation(rhs);
        module.body().append_operation(addi);

        let expected = "module {\n  %c4_i32 = arith.constant 4 : i32\n  %c5_i32 = arith.constant 5 : i32\n  %0 = arith.addi %c4_i32, %c5_i32 : i32\n}\n";
        let output = module.as_operation().to_string();
        assert_eq!(expected, output);
    }

    #[test]
    fn test_create_extsi() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        let src: Operation = create_int_constant(&context, location, Int::I32(4)).unwrap().into();

        let src_value = src.result(0).unwrap().into();
        let result_ty = IntegerType::new(&context, 64).into();
        let extsi = create_extsi(&context, location, src_value, result_ty).unwrap().into();

        module.body().append_operation(src);
        module.body().append_operation(extsi);

        let expected = "module {\n  %c4_i32 = arith.constant 4 : i32\n  %0 = arith.extsi %c4_i32 : i32 to i64\n}\n";
        let output = module.as_operation().to_string();
        assert_eq!(expected, output);
    }

    #[rstest]
    #[case(Predicate::EQ, "eq")]
    #[case(Predicate::NE, "ne")]
    #[case(Predicate::SLT, "slt")]
    #[case(Predicate::SLE, "sle")]
    #[case(Predicate::SGT, "sgt")]
    #[case(Predicate::SGE, "sge")]
    #[case(Predicate::ULT, "ult")]
    #[case(Predicate::ULE, "ule")]
    #[case(Predicate::UGT, "ugt")]
    #[case(Predicate::UGE, "uge")]
    fn test_create_cmpi(#[case] predicate: Predicate, #[case] predicate_str: &str) {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        let lhs: Operation = create_int_constant(&context, location, Int::I32(4)).unwrap().into();
        let rhs: Operation = create_int_constant(&context, location, Int::I32(5)).unwrap().into();

        let lhs_value = lhs.result(0).unwrap().into();
        let rhs_value = rhs.result(0).unwrap().into();
        let result_ty = IntegerType::new(&context, 1).into();

        let cmp_slt = create_cmpi(&context, location, predicate, lhs_value, rhs_value, result_ty)
            .unwrap()
            .into();

        module.body().append_operation(lhs);
        module.body().append_operation(rhs);
        module.body().append_operation(cmp_slt);

        let expected = format!(
            "module {{\n  %c4_i32 = arith.constant 4 : i32\n  %c5_i32 = arith.constant 5 : i32\n  %0 = arith.cmpi {}, %c4_i32, %c5_i32 : i32\n}}\n",
            predicate_str
        );
        let output = module.as_operation().to_string();
        assert_eq!(expected, output);
    }
}
