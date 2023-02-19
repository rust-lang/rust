// edition:2021

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Opcode(pub u8);

impl Opcode {
    pub const OP1: Opcode = Opcode(0x1);
}

pub fn example1(msg_type: Opcode) -> impl FnMut(&[u8]) {
    move |i| match msg_type {
    //~^ ERROR: match is non-exhaustive
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
    //~^ ERROR: match is non-exhaustive
        Opcode2::OP2=> unimplemented!(),
    }
}

fn main() {}
