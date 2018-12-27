#[deny(const_err)]

const FOO: &'static[u32] = &[1, 2, 3];
const BAR: u32 = FOO[5];
//~^ index out of bounds: the len is 3 but the index is 5
//~| ERROR any use of this value will cause an error

fn main() {
    let _ = BAR;
}
