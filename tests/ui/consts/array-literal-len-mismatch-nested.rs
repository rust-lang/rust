// Regression test: when a nested array literal has the wrong length, the
// "consider specifying the actual array length" suggestion must not rewrite the
// length of the outer array type.

const ARR: [[usize; 2]; 2] = [
    [0],
    //~^ ERROR mismatched types
    [1, 2],
];

fn main() {
    let arr: [[usize; 2]; 2] = [
        [0],
        //~^ ERROR mismatched types
        [1, 2],
    ];
}
