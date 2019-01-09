const X: usize = 42 && 39;
//~^ ERROR mismatched types
//~| expected bool, found integer
//~| ERROR mismatched types
//~| expected bool, found integer
//~| ERROR mismatched types
//~| expected usize, found bool
const ARR: [i32; X] = [99; 34];
//~^ ERROR evaluation of constant value failed

const X1: usize = 42 || 39;
//~^ ERROR mismatched types
//~| expected bool, found integer
//~| ERROR mismatched types
//~| expected bool, found integer
//~| ERROR mismatched types
//~| expected usize, found bool
const ARR1: [i32; X1] = [99; 47];
//~^ ERROR evaluation of constant value failed

const X2: usize = -42 || -39;
//~^ ERROR mismatched types
//~| expected bool, found integer
//~| ERROR mismatched types
//~| expected bool, found integer
//~| ERROR mismatched types
//~| expected usize, found bool
const ARR2: [i32; X2] = [99; 18446744073709551607];
//~^ ERROR evaluation of constant value failed

const X3: usize = -42 && -39;
//~^ ERROR mismatched types
//~| expected bool, found integer
//~| ERROR mismatched types
//~| expected bool, found integer
//~| ERROR mismatched types
//~| expected usize, found bool
const ARR3: [i32; X3] = [99; 6];
//~^ ERROR evaluation of constant value failed

const Y: usize = 42.0 == 42.0;
//~^ ERROR mismatched types
//~| expected usize, found bool
const ARRR: [i32; Y] = [99; 1];
//~^ ERROR evaluation of constant value failed

const Y1: usize = 42.0 >= 42.0;
//~^ ERROR mismatched types
//~| expected usize, found bool
const ARRR1: [i32; Y1] = [99; 1];
//~^ ERROR evaluation of constant value failed

const Y2: usize = 42.0 <= 42.0;
//~^ ERROR mismatched types
//~| expected usize, found bool
const ARRR2: [i32; Y2] = [99; 1];
//~^ ERROR evaluation of constant value failed

const Y3: usize = 42.0 > 42.0;
//~^ ERROR mismatched types
//~| expected usize, found bool
const ARRR3: [i32; Y3] = [99; 0];
//~^ ERROR evaluation of constant value failed

const Y4: usize = 42.0 < 42.0;
//~^ ERROR mismatched types
//~| expected usize, found bool
const ARRR4: [i32; Y4] = [99; 0];
//~^ ERROR evaluation of constant value failed

const Y5: usize = 42.0 != 42.0;
//~^ ERROR mismatched types
//~| expected usize, found bool
const ARRR5: [i32; Y5] = [99; 0];
//~^ ERROR evaluation of constant value failed

fn main() {
    let _ = ARR;
    let _ = ARRR;
    let _ = ARRR1;
    let _ = ARRR2;
    let _ = ARRR3;
    let _ = ARRR4;
    let _ = ARRR5;
}
