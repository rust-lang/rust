const fn mutable_ref_in_const() -> u8 {
    let mut a = 0;
    let b = &mut a; //~ ERROR mutable references in const fn
    *b
}

struct X;

impl X {
    const fn inherent_mutable_ref_in_const() -> u8 {
        let mut a = 0;
        let b = &mut a; //~ ERROR mutable references in const fn
        *b
    }
}

fn main() {}
