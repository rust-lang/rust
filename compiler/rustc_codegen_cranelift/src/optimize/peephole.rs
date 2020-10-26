//! Peephole optimizations that can be performed while creating clif ir.

use cranelift_codegen::ir::{
    condcodes::IntCC, types, InstBuilder, InstructionData, Opcode, Value, ValueDef,
};
use cranelift_frontend::FunctionBuilder;

/// If the given value was produced by a `bint` instruction, return it's input, otherwise return the
/// given value.
pub(crate) fn maybe_unwrap_bint(bcx: &mut FunctionBuilder<'_>, arg: Value) -> Value {
    if let ValueDef::Result(arg_inst, 0) = bcx.func.dfg.value_def(arg) {
        match bcx.func.dfg[arg_inst] {
            InstructionData::Unary {
                opcode: Opcode::Bint,
                arg,
            } => arg,
            _ => arg,
        }
    } else {
        arg
    }
}

/// If the given value was produced by the lowering of `Rvalue::Not` return the input and true,
/// otherwise return the given value and false.
pub(crate) fn maybe_unwrap_bool_not(bcx: &mut FunctionBuilder<'_>, arg: Value) -> (Value, bool) {
    if let ValueDef::Result(arg_inst, 0) = bcx.func.dfg.value_def(arg) {
        match bcx.func.dfg[arg_inst] {
            // This is the lowering of `Rvalue::Not`
            InstructionData::IntCompareImm {
                opcode: Opcode::IcmpImm,
                cond: IntCC::Equal,
                arg,
                imm,
            } if imm.bits() == 0 => (arg, true),
            _ => (arg, false),
        }
    } else {
        (arg, false)
    }
}

pub(crate) fn make_branchable_value(bcx: &mut FunctionBuilder<'_>, arg: Value) -> Value {
    if bcx.func.dfg.value_type(arg).is_bool() {
        return arg;
    }

    (|| {
        let arg_inst = if let ValueDef::Result(arg_inst, 0) = bcx.func.dfg.value_def(arg) {
            arg_inst
        } else {
            return None;
        };

        match bcx.func.dfg[arg_inst] {
            // This is the lowering of Rvalue::Not
            InstructionData::Load {
                opcode: Opcode::Load,
                arg: ptr,
                flags,
                offset,
            } => {
                // Using `load.i8 + uextend.i32` would legalize to `uload8 + ireduce.i8 +
                // uextend.i32`. Just `uload8` is much faster.
                match bcx.func.dfg.ctrl_typevar(arg_inst) {
                    types::I8 => Some(bcx.ins().uload8(types::I32, flags, ptr, offset)),
                    types::I16 => Some(bcx.ins().uload16(types::I32, flags, ptr, offset)),
                    _ => None,
                }
            }
            _ => None,
        }
    })()
    .unwrap_or_else(|| {
        match bcx.func.dfg.value_type(arg) {
            types::I8 | types::I32 => {
                // WORKAROUND for brz.i8 and brnz.i8 not yet being implemented
                bcx.ins().uextend(types::I32, arg)
            }
            _ => arg,
        }
    })
}
