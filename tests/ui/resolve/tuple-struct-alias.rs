struct S(u8, u16);
type A = S;

fn main() {
    let s = A(0, 1); //~ ERROR expected function
    match s {
        A(..) => {} //~ ERROR expected tuple struct or tuple variant
    }
}
