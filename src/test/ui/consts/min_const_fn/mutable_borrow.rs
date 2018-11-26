const fn mutable_ref_in_const() -> u8 {
    let mut a = 0; //~ ERROR local variables in const fn
    let b = &mut a;
    *b
}

struct X;

impl X {
    const fn inherent_mutable_ref_in_const() -> u8 {
        let mut a = 0; //~ ERROR local variables in const fn
        let b = &mut a;
        *b
    }
}

fn main() {}
