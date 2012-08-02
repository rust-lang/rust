fn f() -> int {
    let mut x: int;
    while 1 == 1 { x = 10; }
    return x; //~ ERROR use of possibly uninitialized variable: `x`
}

fn main() { f(); }
