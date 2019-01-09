// Tests that the use of uninitialized variable in assignment operator
// expression is detected.

pub fn main() {
    let x: isize;
    x += 1; //~ ERROR use of possibly uninitialized variable: `x`

    let x: isize;
    x -= 1; //~ ERROR use of possibly uninitialized variable: `x`

    let x: isize;
    x *= 1; //~ ERROR use of possibly uninitialized variable: `x`

    let x: isize;
    x /= 1; //~ ERROR use of possibly uninitialized variable: `x`

    let x: isize;
    x %= 1; //~ ERROR use of possibly uninitialized variable: `x`

    let x: isize;
    x ^= 1; //~ ERROR use of possibly uninitialized variable: `x`

    let x: isize;
    x &= 1; //~ ERROR use of possibly uninitialized variable: `x`

    let x: isize;
    x |= 1; //~ ERROR use of possibly uninitialized variable: `x`

    let x: isize;
    x <<= 1;    //~ ERROR use of possibly uninitialized variable: `x`

    let x: isize;
    x >>= 1;    //~ ERROR use of possibly uninitialized variable: `x`
}
