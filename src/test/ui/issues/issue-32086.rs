struct S(u8);
const C: S = S(10);

fn main() {
    let C(a) = S(11); //~ ERROR expected tuple struct/variant, found constant `C`
    let C(..) = S(11); //~ ERROR expected tuple struct/variant, found constant `C`
}
