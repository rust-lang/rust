// test for #112824 ICE type mismatching when copying!

pub struct Opcode(pub u8);

pub struct Opcode2(&'a S);
//~^ ERROR use of undeclared lifetime name `'a`
//~^^ ERROR cannot find type `S` in this scope

impl Opcode2 {
    pub const OP2: Opcode2 = Opcode2(Opcode(0x1));
}

pub fn example2(msg_type: Opcode2) -> impl FnMut(&[u8]) {
    move |i| match msg_type {
        Opcode2::OP2 => unimplemented!(), // ok, `const` already emitted an error
    }
}

pub fn main() {}
