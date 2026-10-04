struct S(u8, u16);
type A = S;

fn main() {
    let s = A(0, 1); //~ ERROR cannot find function, tuple struct or tuple variant `A` in this scope
    match s {
        A(..) => {} //~ ERROR cannot find tuple struct or tuple variant `A` in this scope
    }
}
