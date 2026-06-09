fn main() {
    let a: u16;
    let b: u16 = 42;
    let c: usize = 5;

    a = c + b * 5; //~ ERROR: mismatched types [E0308]
                   //~| ERROR: mismatched types [E0308]
                   //~| ERROR: cannot add `u16` to `usize` [E0277]
}
