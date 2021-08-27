// edition:2021

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Opcode(pub u8);

impl Opcode {
    pub const OP1: Opcode = Opcode(0x1);
}

pub fn example(msg_type: Opcode) -> impl FnMut(&[u8]) {
    move |i| match msg_type {
    //~^ ERROR: non-exhaustive patterns: `Opcode(0_u8)` and `Opcode(2_u8..=u8::MAX)` not covered
        Opcode::OP1 => unimplemented!(),
    }
}

#[derive(Debug)]
enum V {
    Single(i32, i32)
}

fn valid1() {
    let mut v = V::Single(0, 0);
    let mv = &mut v;

    match v {
        V::Single(..) => println!("asd"),
    };

    *mv  = V::Single(0, 1);
}

fn valid2() {
    let mut v = V::Single(0, 0);
    let mv = &mut v;

    (|| match v {
        V::Single(..) => println!("asd"),
    })();

    *mv  = V::Single(0, 1);
}

fn main() {
    valid1();
    valid2();
}
