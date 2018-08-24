struct S(u8, u16);
type A = S;

impl S {
    fn f() {
        let s = Self(0, 1); //~ ERROR expected function
        match s {
            Self(..) => {} //~ ERROR expected tuple struct/variant
        }
    }
}

fn main() {
    let s = A(0, 1); //~ ERROR expected function
    match s {
        A(..) => {} //~ ERROR expected tuple struct/variant
    }
}
