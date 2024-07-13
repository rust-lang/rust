//@ check-pass

const fn const_address_of_in_const() {
    let mut a = 0;
    let b = &raw const a;
}

struct X;

impl X {
    const fn inherent_const_address_of_in_const() {
        let mut a = 0;
        let b = &raw const a;
    }
}

fn main() {}
