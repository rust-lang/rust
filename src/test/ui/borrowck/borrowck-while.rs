fn f() -> isize {
    let mut x: isize;
    while 1 == 1 { x = 10; }
    return x; //~ ERROR use of possibly uninitialized variable: `x`
}

fn main() { f(); }
