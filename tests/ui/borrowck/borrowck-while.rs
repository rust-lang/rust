fn f() -> isize {
    let mut x: isize;
    while 1 == 1 { x = 10; }
    return x; //~ ERROR E0381
}

fn main() { f(); }
