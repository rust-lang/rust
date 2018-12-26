// Test that negating unsigned integers doesn't compile

struct S;
impl std::ops::Neg for S {
    type Output = u32;
    fn neg(self) -> u32 { 0 }
}

fn main() {
    let _max: usize = -1;
    //~^ ERROR cannot apply unary operator `-` to type `usize`

    let x = 5u8;
    let _y = -x;
    //~^ ERROR cannot apply unary operator `-` to type `u8`

    -S; // should not trigger the gate; issue 26840
}
