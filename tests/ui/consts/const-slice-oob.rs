const FOO: &'static [u32] = &[1, 2, 3];
const BAR: u32 = FOO[5];
//~^ ERROR index out of bounds: the length is 3 but the index is 5

fn main() {
    let _ = BAR;
}
