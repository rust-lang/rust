struct S(u8, u16);

fn main() {
    let s = S{0b1: 10, 0: 11};
    //~^ ERROR struct `S` has no field named `0b1`
    match s {
        S{0: a, 0x1: b, ..} => {}
        //~^ ERROR does not have a field named `0x1`
    }
}
