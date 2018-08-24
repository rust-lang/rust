fn test() {
    let mut v: isize;
    v = v + 1; //~ ERROR use of possibly uninitialized variable: `v`
    v.clone();
}

fn main() {
}
