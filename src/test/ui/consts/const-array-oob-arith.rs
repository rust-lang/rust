#![feature(const_indexing)]

const ARR: [i32; 6] = [42, 43, 44, 45, 46, 47];
const IDX: usize = 3;
const VAL: i32 = ARR[IDX];
const BONG: [i32; (ARR[0] - 41) as usize] = [5];
const BLUB: [i32; (ARR[0] - 40) as usize] = [5];
//~^ ERROR: mismatched types
//~| expected an array with a fixed size of 2 elements, found one with 1 element
const BOO: [i32; (ARR[0] - 41) as usize] = [5, 99];
//~^ ERROR: mismatched types
//~| expected an array with a fixed size of 1 element, found one with 2 elements

fn main() {
    let _ = VAL;
}
