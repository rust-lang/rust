fn test() {
    let v: isize;
    v += 1; //~ ERROR use of possibly uninitialized variable: `v`
    v.clone();
}

fn main() {
}
