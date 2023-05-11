//! Defines [`Pointer`] which is used to improve the quality of the generated clif ir for pointer
//! operations.

use crate::prelude::*;

use rustc_target::abi::Align;

use cranelift_codegen::ir::immediates::Offset32;

/// A pointer pointing either to a certain address, a certain stack slot or nothing.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Pointer {
    base: PointerBase,
    offset: Offset32,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum PointerBase {
    Addr(Value),
    Stack(StackSlot),
    Dangling(Align),
}

impl Pointer {
    pub(crate) fn new(addr: Value) -> Self {
        Pointer { base: PointerBase::Addr(addr), offset: Offset32::new(0) }
    }

    pub(crate) fn stack_slot(stack_slot: StackSlot) -> Self {
        Pointer { base: PointerBase::Stack(stack_slot), offset: Offset32::new(0) }
    }

    pub(crate) fn dangling(align: Align) -> Self {
        Pointer { base: PointerBase::Dangling(align), offset: Offset32::new(0) }
    }

    pub(crate) fn debug_base_and_offset(self) -> (PointerBase, Offset32) {
        (self.base, self.offset)
    }

    pub(crate) fn get_addr(self, fx: &mut FunctionCx<'_, '_, '_>) -> Value {
        match self.base {
            PointerBase::Addr(base_addr) => {
                let offset: i64 = self.offset.into();
                if offset == 0 { base_addr } else { fx.bcx.ins().iadd_imm(base_addr, offset) }
            }
            PointerBase::Stack(stack_slot) => {
                fx.bcx.ins().stack_addr(fx.pointer_type, stack_slot, self.offset)
            }
            PointerBase::Dangling(align) => {
                fx.bcx.ins().iconst(fx.pointer_type, i64::try_from(align.bytes()).unwrap())
            }
        }
    }

    pub(crate) fn offset(self, fx: &mut FunctionCx<'_, '_, '_>, extra_offset: Offset32) -> Self {
        self.offset_i64(fx, extra_offset.into())
    }

    pub(crate) fn offset_i64(self, fx: &mut FunctionCx<'_, '_, '_>, extra_offset: i64) -> Self {
        if let Some(new_offset) = self.offset.try_add_i64(extra_offset) {
            Pointer { base: self.base, offset: new_offset }
        } else {
            let base_offset: i64 = self.offset.into();
            if let Some(new_offset) = base_offset.checked_add(extra_offset) {
                let base_addr = match self.base {
                    PointerBase::Addr(addr) => addr,
                    PointerBase::Stack(stack_slot) => {
                        fx.bcx.ins().stack_addr(fx.pointer_type, stack_slot, 0)
                    }
                    PointerBase::Dangling(align) => {
                        fx.bcx.ins().iconst(fx.pointer_type, i64::try_from(align.bytes()).unwrap())
                    }
                };
                let addr = fx.bcx.ins().iadd_imm(base_addr, new_offset);
                Pointer { base: PointerBase::Addr(addr), offset: Offset32::new(0) }
            } else {
                panic!(
                    "self.offset ({}) + extra_offset ({}) not representable in i64",
                    base_offset, extra_offset
                );
            }
        }
    }

    pub(crate) fn offset_value(self, fx: &mut FunctionCx<'_, '_, '_>, extra_offset: Value) -> Self {
        match self.base {
            PointerBase::Addr(addr) => Pointer {
                base: PointerBase::Addr(fx.bcx.ins().iadd(addr, extra_offset)),
                offset: self.offset,
            },
            PointerBase::Stack(stack_slot) => {
                let base_addr = fx.bcx.ins().stack_addr(fx.pointer_type, stack_slot, self.offset);
                Pointer {
                    base: PointerBase::Addr(fx.bcx.ins().iadd(base_addr, extra_offset)),
                    offset: Offset32::new(0),
                }
            }
            PointerBase::Dangling(align) => {
                let addr =
                    fx.bcx.ins().iconst(fx.pointer_type, i64::try_from(align.bytes()).unwrap());
                Pointer {
                    base: PointerBase::Addr(fx.bcx.ins().iadd(addr, extra_offset)),
                    offset: self.offset,
                }
            }
        }
    }

    pub(crate) fn load(self, fx: &mut FunctionCx<'_, '_, '_>, ty: Type, flags: MemFlags) -> Value {
        match self.base {
            PointerBase::Addr(base_addr) => fx.bcx.ins().load(ty, flags, base_addr, self.offset),
            PointerBase::Stack(stack_slot) => fx.bcx.ins().stack_load(ty, stack_slot, self.offset),
            PointerBase::Dangling(_align) => unreachable!(),
        }
    }

    pub(crate) fn store(self, fx: &mut FunctionCx<'_, '_, '_>, value: Value, flags: MemFlags) {
        match self.base {
            PointerBase::Addr(base_addr) => {
                fx.bcx.ins().store(flags, value, base_addr, self.offset);
            }
            PointerBase::Stack(stack_slot) => {
                fx.bcx.ins().stack_store(value, stack_slot, self.offset);
            }
            PointerBase::Dangling(_align) => unreachable!(),
        }
    }
}
