#[deny(const_err)]

const FOO: &'static[u32] = &[1, 2, 3];
const BAR: u32 = FOO[5];
//~^ index out of bounds: the length is 3 but the index is 5
//~| ERROR any use of this value will cause an error
//~| WARN this was previously accepted by the compiler but is being phased out

fn main() {
    let _ = BAR;
}
