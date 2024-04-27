fn f() -> isize {
    let mut x: isize;
    for _ in 0..0 { x = 10; }
    return x; //~ ERROR E0381
}

fn main() { f(); }
