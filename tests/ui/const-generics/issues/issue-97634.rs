//@ build-pass

pub enum Register<const N: u16> {
    Field0 = 40,
    Field1,
}

fn main() {
    let _b = Register::<0>::Field1 as u16;
}
