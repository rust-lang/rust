const FOO: [usize; 3] = [1, 2, 3];
const BAR: usize = FOO[5];
//~^ ERROR: index out of bounds: the length is 3 but the index is 5

const BLUB: [u32; FOO[4]] = [5, 6];
//~^ ERROR index out of bounds: the length is 3 but the index is 4

fn main() {
    let _ = BAR;
}
