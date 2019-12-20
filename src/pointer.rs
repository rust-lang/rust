use crate::prelude::*;

use cranelift::codegen::ir::immediates::Offset32;

#[derive(Copy, Clone, Debug)]
pub struct Pointer {
    base_addr: Value,
    offset: Offset32,
}

impl Pointer {
    pub fn new(addr: Value) -> Self {
        Pointer {
            base_addr: addr,
            offset: Offset32::new(0),
        }
    }

    pub fn const_addr<'a, 'tcx>(fx: &mut FunctionCx<'a, 'tcx, impl Backend>, addr: i64) -> Self {
        let addr = fx.bcx.ins().iconst(fx.pointer_type, addr);
        Pointer {
            base_addr: addr,
            offset: Offset32::new(0),
        }
    }

    pub fn get_addr<'a, 'tcx>(self, fx: &mut FunctionCx<'a, 'tcx, impl Backend>) -> Value {
        let offset: i64 = self.offset.into();
        if offset == 0 {
            self.base_addr
        } else {
            fx.bcx.ins().iadd_imm(self.base_addr, offset)
        }
    }

    pub fn get_addr_and_offset(self) -> (Value, Offset32) {
        (self.base_addr, self.offset)
    }

    pub fn offset<'a, 'tcx>(
        self,
        fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
        extra_offset: Offset32,
    ) -> Self {
        self.offset_i64(fx, extra_offset.into())
    }

    pub fn offset_i64<'a, 'tcx>(
        self,
        fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
        extra_offset: i64,
    ) -> Self {
        if let Some(new_offset) = self.offset.try_add_i64(extra_offset) {
            Pointer {
                base_addr: self.base_addr,
                offset: new_offset,
            }
        } else {
            let base_offset: i64 = self.offset.into();
            if let Some(new_offset) = base_offset.checked_add(extra_offset){
                let addr = fx.bcx.ins().iadd_imm(self.base_addr, new_offset);
                Pointer {
                    base_addr: addr,
                    offset: Offset32::new(0),
                }
            } else {
                panic!("self.offset ({}) + extra_offset ({}) not representable in i64", base_offset, extra_offset);
            }
        }
    }

    pub fn offset_value<'a, 'tcx>(
        self,
        fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
        extra_offset: Value,
    ) -> Self {
        let base_addr = fx.bcx.ins().iadd(self.base_addr, extra_offset);
        Pointer {
            base_addr,
            offset: self.offset,
        }
    }

    pub fn load<'a, 'tcx>(
        self,
        fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
        ty: Type,
        flags: MemFlags,
    ) -> Value {
        fx.bcx.ins().load(ty, flags, self.base_addr, self.offset)
    }

    pub fn store<'a, 'tcx>(
        self,
        fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
        value: Value,
        flags: MemFlags,
    ) {
        fx.bcx.ins().store(flags, value, self.base_addr, self.offset);
    }
}
