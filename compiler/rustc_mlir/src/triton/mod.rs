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
use melior::ir::attribute::{
    ArrayAttribute, Attribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute,
    TypeAttribute,
};
use melior::ir::operation::OperationBuilder;
use melior::ir::r#type::{FunctionType, IntegerType};
use melior::ir::{Identifier, Location, Operation, Region, Type, TypeLike, Value};

use crate::errors::Error;
use crate::ffi::{mlirCreateTritonPointerType, mlirLoadTritonDialect};
use crate::triton::tt::{CallOperation, FuncOperation, IntToPtrOperation, PtrToIntOperation, ReturnOperation};

pub mod compiler;
pub mod program;
pub mod tensor;

pub use compiler::TritonCompiler;

melior_macro::dialect! {
    name: "tt",
    files: [
        "triton/Dialect/Triton/IR/TritonDialect.td",
        "triton/Dialect/Triton/IR/TritonOps.td",
        "triton/Dialect/Triton/IR/TritonTypes.td"
    ],
    include_directories: ["TRITON_INCLUDE_DIRECTORY"],
}

pub fn load_triton_dialect(context: &Context) {
    unsafe {
        mlirLoadTritonDialect(context.to_raw());
    }
}

pub fn attr_i32<'ctx>(context: &'ctx Context, value: i32) -> IntegerAttribute<'ctx> {
    IntegerAttribute::new(IntegerType::new(context, 32).into(), value as i64)
}

pub fn pointer_type<'a>(pointee: Type<'a>) -> Type<'a> {
    unsafe { Type::from_raw(mlirCreateTritonPointerType(pointee.to_raw(), 1)) }
}

// Extracted function for creating a tt.func operation with empty body and tt.divisibility attrs.
pub fn create_func<'c>(
    context: &'c melior::Context,
    location: Location<'c>,
    name: &str,
    visibility: &str,
    arg_types: &[Type<'c>],
    res_types: &[Type<'c>],
    divisibility: i32,
) -> Result<FuncOperation<'c>, Error> {
    // Create the function type
    let function_type = TypeAttribute::new(FunctionType::new(context, arg_types, res_types).into());

    // Argument attributes: each gets a dictionary {tt.divisibility = $divisibility : i32}
    let arg_attrs: Vec<_> = (0..arg_types.len())
        .map(|_| {
            Attribute::parse(context, &format!("{{tt.divisibility = {} : i32}}", divisibility))
                .expect("valid arg attrs")
        })
        .collect();
    let arg_attrs = ArrayAttribute::new(context, &arg_attrs);

    // Result attributes: empty dict for each return type (can be generalised, but empty suffices for now)
    let res_attrs: Vec<_> =
        (0..res_types.len()).map(|_| Attribute::parse(context, "{}").unwrap()).collect();
    let res_attrs = ArrayAttribute::new(context, &res_attrs);

    // Empty function body
    let body_region = Region::new();

    Ok(FuncOperation::builder(context, location)
        .sym_name(StringAttribute::new(context, name))
        .function_type(function_type)
        .sym_visibility(StringAttribute::new(context, visibility))
        .arg_attrs(arg_attrs)
        .res_attrs(res_attrs)
        .body(body_region)
        .build())
}

pub fn create_return<'ctx, 'b>(
    context: &'ctx Context,
    location: Location<'ctx>,
    value: &[Value<'ctx, 'b>],
) -> Result<ReturnOperation<'ctx>, Error> {
    Ok(ReturnOperation::builder(context, location).srcs(value).build())
}

pub fn call<'ctx>(
    context: &'ctx Context,
    location: Location<'ctx>,
    callee: &str,
    args: &[Value<'ctx, 'ctx>],
    result_ty: &[Type<'ctx>],
) -> Result<CallOperation<'ctx>, Error> {
    let callee_attr = FlatSymbolRefAttribute::new(context, callee);

    // Attach empty dictionaries for each argument/result to satisfy the
    // `ArgAndResultAttrsOpInterface` requirements of `tt.call`.
    let arg_attr_dicts: Vec<_> = (0..args.len())
        .map(|_| Attribute::parse(context, "{}").expect("valid empty arg attrs dict"))
        .collect();
    let arg_attrs = ArrayAttribute::new(context, &arg_attr_dicts);

    let res_attr_dicts: Vec<_> = (0..result_ty.len())
        .map(|_| Attribute::parse(context, "{}").expect("valid empty res attrs dict"))
        .collect();
    let res_attrs = ArrayAttribute::new(context, &res_attr_dicts);

    // `CallOperation` has `Variadic<AnyType>` results without an explicit ODS name,
    // so the melior-generated typed builder does not expose a result-type setter.
    // We therefore fall back to the raw `OperationBuilder`, which accepts result
    // types via `add_results`, and then coerce the result into `CallOperation`.
    let op = OperationBuilder::new("tt.call", location)
        .add_attributes(&[
            (Identifier::new(context, "callee"), callee_attr.into()),
            (Identifier::new(context, "arg_attrs"), arg_attrs.into()),
            (Identifier::new(context, "res_attrs"), res_attrs.into()),
        ])
        .add_operands(args)
        .add_results(result_ty)
        .build()
        .map_err(|e| Error::InvalidType { msg: format!("failed to build tt.call: {e}") })?;

    op.try_into().map_err(|_| Error::InvalidType { msg: "tt.call operation type mismatch".into() })
}

pub fn int_to_ptr<'ctx, 'b>(
    context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, 'b>,
    dest: Type<'ctx>,
) -> Result<IntToPtrOperation<'ctx>, Error> {
    Ok(IntToPtrOperation::builder(context, location).result(dest).src(src).build())
}

pub fn ptr_to_int<'ctx, 'b>(
    context: &'ctx Context,
    location: Location<'ctx>,
    src: Value<'ctx, 'b>,
    dest: Type<'ctx>,
) -> Result<PtrToIntOperation<'ctx>, Error> {
    Ok(PtrToIntOperation::builder(context, location).result(dest).src(src).build())
}

#[cfg(test)]
mod tests {
    use melior::dialect::ods::arith;
    use melior::ir::attribute::BoolAttribute;
    use melior::ir::operation::{OperationLike, OperationMutLike};
    use melior::ir::{Block, BlockLike, Location, Module, Operation, RegionLike, Type, Value};

    use super::*;
    use crate::shared::arith::{Int, create_int_constant};
    use crate::test::create_test_context;

    #[test]
    fn test_func_op_with_attributes() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = Type::float32(&context);
        let ptr_f32_type = pointer_type(f32_type);

        // Function signature: (!tt.ptr<f32>, !tt.ptr<f32>) -> f32
        let inputs = vec![ptr_f32_type, ptr_f32_type];
        let results = vec![f32_type];

        // Use the new function in test:
        let func_op =
            create_func(&context, location, "test_func_attrs", "public", &inputs, &results, 16)
                .unwrap();

        // Create a constant op returning f32 1.0
        let one_attr = Attribute::parse(&context, "1.0 : f32").expect("valid f32");
        let const_op = arith::ConstantOperation::builder(&context, location)
            .value(one_attr)
            .result(f32_type)
            .build();

        // Return from triton func with tt.return
        let return_op = ReturnOperation::builder(&context, location)
            .srcs(&[const_op.result().unwrap().into()])
            .build();

        // Insert block into function body region
        let first_block = Block::new(&[(ptr_f32_type, location), (ptr_f32_type, location)]);
        first_block.append_operation(const_op.into());
        first_block.append_operation(return_op.into());
        func_op.body().unwrap().append_block(first_block);

        // Add noinline attribute to the function operation
        let mut func_op: Operation = func_op.into();
        func_op.set_attribute("noinline", BoolAttribute::new(&context, false).into());

        module.body().append_operation(func_op);

        let output = module.as_operation().to_string();

        let expected = "module {
  tt.func public @test_func_attrs(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> f32 attributes {noinline = false} {
    %cst = arith.constant 1.000000e+00 : f32
    tt.return %cst : f32
  }
}
";
        assert_eq!(expected, output);
    }

    #[test]
    fn test_create_int_to_ptr() {
        // MLIR context and location
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        // f32 type as MLIR type
        let f32_type = Type::float32(&context);
        let f32_ptr_type = pointer_type(f32_type);
        let i64_zero = create_int_constant(&context, location, Int::I64(0)).unwrap();
        let i64_zero_value = i64_zero.result().unwrap();

        // Call the function under test
        let cast_op = int_to_ptr(&context, location, i64_zero_value.into(), f32_ptr_type).unwrap();

        module.body().append_operation(i64_zero.into());
        module.body().append_operation(cast_op.into());

        let output = module.as_operation().to_string();

        let expected = "module {
  %c0_i64 = arith.constant 0 : i64
  %0 = tt.int_to_ptr %c0_i64 : i64 -> !tt.ptr<f32>
}
";
        assert_eq!(expected, output);
    }

    /// Verify `tt.call` emits the correct MLIR form.
    ///
    /// The assembly format for `tt.call` is:
    ///   `$callee ( $operands ) attr-dict : functional-type($operands, results)`
    ///
    /// Empty `arg_attrs`/`res_attrs` are elided from the attr-dict by MLIR's printer
    /// (matching real Triton test files), so the expected form is:
    ///   `%0 = tt.call @my_add(%arg0, %arg1) : (f32, f32) -> f32`
    #[test]
    fn test_call_op() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f32_type = Type::float32(&context);

        // ── callee: my_add(%arg0: f32, %arg1: f32) -> f32 ──────────────────────
        let callee_func =
            create_func(&context, location, "my_add", "private", &[f32_type, f32_type], &[f32_type], 16)
                .unwrap();
        let callee_block = Block::new(&[(f32_type, location), (f32_type, location)]);
        let first_arg: Value<'_, '_> = callee_block.argument(0).unwrap().into();
        let callee_return = create_return(&context, location, &[first_arg]).unwrap();
        callee_block.append_operation(callee_return.into());
        callee_func.body().unwrap().append_block(callee_block);
        module.body().append_operation(callee_func.into());

        // ── caller: calls my_add with block arguments ───────────────────────────
        let caller_func =
            create_func(&context, location, "caller", "public", &[f32_type, f32_type], &[f32_type], 16)
                .unwrap();
        let caller_block = Block::new(&[(f32_type, location), (f32_type, location)]);
        let arg0: Value<'_, '_> = caller_block.argument(0).unwrap().into();
        let arg1: Value<'_, '_> = caller_block.argument(1).unwrap().into();

        let call_op: Operation<'_> =
            call(&context, location, "my_add", &[arg0, arg1], &[f32_type]).unwrap().into();
        let call_result: Value<'_, '_> = call_op.result(0).unwrap().into();
        let caller_return = create_return(&context, location, &[call_result]).unwrap();
        caller_block.append_operation(call_op);
        caller_block.append_operation(caller_return.into());
        caller_func.body().unwrap().append_block(caller_block);
        module.body().append_operation(caller_func.into());

        let output = module.as_operation().to_string();

        // Verify the call op emits the correct mnemonic, callee symbol, and
        // functional type.  We use contains() rather than exact matching to remain
        // robust to how MLIR renders the optional/empty arg_attrs/res_attrs.
        assert!(
            output.contains("tt.call @my_add"),
            "expected 'tt.call @my_add' in output:\n{output}"
        );
        assert!(
            output.contains(": (f32, f32) -> f32"),
            "expected functional-type '(f32, f32) -> f32' in output:\n{output}"
        );
    }

    /// Verify `tt.call` with no arguments and no results.
    ///
    /// Assembly form: `tt.call @my_void_func() : () -> ()`
    #[test]
    fn test_call_op_void() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        // ── callee: my_void_func() -> () ────────────────────────────────────────
        let callee_func =
            create_func(&context, location, "my_void_func", "private", &[], &[], 16).unwrap();
        let callee_block = Block::new(&[]);
        let callee_return = create_return(&context, location, &[]).unwrap();
        callee_block.append_operation(callee_return.into());
        callee_func.body().unwrap().append_block(callee_block);
        module.body().append_operation(callee_func.into());

        // ── caller: calls my_void_func with no args ──────────────────────────────
        let caller_func =
            create_func(&context, location, "caller_void", "public", &[], &[], 16).unwrap();
        let caller_block = Block::new(&[]);
        let void_call = call(&context, location, "my_void_func", &[], &[]).unwrap();
        let void_return = create_return(&context, location, &[]).unwrap();
        caller_block.append_operation(void_call.into());
        caller_block.append_operation(void_return.into());
        caller_func.body().unwrap().append_block(caller_block);
        module.body().append_operation(caller_func.into());

        let output = module.as_operation().to_string();

        assert!(
            output.contains("tt.call @my_void_func()"),
            "expected 'tt.call @my_void_func()' in output:\n{output}"
        );
        assert!(
            output.contains(": () -> ()"),
            "expected functional-type '() -> ()' in output:\n{output}"
        );
    }

    #[test]
    fn test_create_ptr_to_int() {
        let context = create_test_context();
        load_triton_dialect(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        // Create a pointer value via int_to_ptr, then cast it back to i64
        let f32_type = Type::float32(&context);
        let f32_ptr_type = pointer_type(f32_type);
        let i64_type = Type::from(IntegerType::new(&context, 64));

        let i64_zero = create_int_constant(&context, location, Int::I64(0)).unwrap();
        let i64_zero_value = i64_zero.result().unwrap();

        let int_to_ptr_op =
            int_to_ptr(&context, location, i64_zero_value.into(), f32_ptr_type).unwrap();
        let ptr_value: Value<'_, '_> = int_to_ptr_op.result().unwrap().into();

        let cast_op = ptr_to_int(&context, location, ptr_value, i64_type).unwrap();

        module.body().append_operation(i64_zero.into());
        module.body().append_operation(int_to_ptr_op.into());
        module.body().append_operation(cast_op.into());

        let output = module.as_operation().to_string();

        let expected = "module {
  %c0_i64 = arith.constant 0 : i64
  %0 = tt.int_to_ptr %c0_i64 : i64 -> !tt.ptr<f32>
  %1 = tt.ptr_to_int %0 : !tt.ptr<f32> -> i64
}
";
        assert_eq!(expected, output);
    }
}
