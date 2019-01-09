fn f() -> isize {
    let x: isize;
    return x; //~ ERROR use of possibly uninitialized variable: `x`
}

fn main() { f(); }
