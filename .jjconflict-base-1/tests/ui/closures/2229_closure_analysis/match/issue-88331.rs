//@ edition:2021

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Opcode(pub u8);

impl Opcode {
    pub const OP1: Opcode = Opcode(0x1);
}

pub fn example1(msg_type: Opcode) -> impl FnMut(&[u8]) {
    move |i| match msg_type {
    //~^ ERROR: non-exhaustive patterns: `Opcode(0_u8)` and `Opcode(2_u8..=u8::MAX)` not covered
        Opcode::OP1 => unimplemented!(),
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Opcode2(Opcode);

impl Opcode2 {
    pub const OP2: Opcode2 = Opcode2(Opcode(0x1));
}


pub fn example2(msg_type: Opcode2) -> impl FnMut(&[u8]) {

    move |i| match msg_type {
    //~^ ERROR: non-exhaustive patterns: `Opcode2(Opcode(0_u8))` and `Opcode2(Opcode(2_u8..=u8::MAX))` not covered
        Opcode2::OP2=> unimplemented!(),
    }
}

fn main() {}
