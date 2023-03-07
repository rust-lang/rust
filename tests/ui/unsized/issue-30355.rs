pub struct X([u8]);

pub static Y: &'static X = {
    const Y: &'static [u8] = b"";
    &X(*Y)
    //~^ ERROR E0277
};

fn main() {}
