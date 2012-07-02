fn test() {
    let w: ~[int];
    w[5] = 0; //~ ERROR use of possibly uninitialized variable: `w`
}

fn main() { test(); }
