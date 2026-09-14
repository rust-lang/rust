struct S(i32);
enum E { V(i32) }

fn main() {
    let S {} = S(0); //~ ERROR tuple variant `S` written as struct variant
    let E::V {} = E::V(0); //~ ERROR tuple variant `E::V` written as struct variant
}
