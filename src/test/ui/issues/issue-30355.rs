pub struct X([u8]);

pub static Y: &'static X = {
    const Y: &'static [u8] = b"";
    &X(*Y)
    //~^ ERROR cannot move out
    //~^^ ERROR cannot move a
    //~^^^ ERROR cannot move a
};

fn main() {}
