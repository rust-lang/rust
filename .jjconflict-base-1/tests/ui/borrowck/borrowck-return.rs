fn f() -> isize {
    let x: isize;
    return x; //~ ERROR E0381
}

fn main() { f(); }
