const FOO: [usize; 3] = [1, 2, 3];
const BAR: usize = FOO[5];
//~^ ERROR: evaluation of constant value failed

const BLUB: [u32; FOO[4]] = [5, 6];
//~^ ERROR evaluation of constant value failed [E0080]
//~| index out of bounds: the length is 3 but the index is 4

fn main() {
    let _ = BAR;
}
