fn test() {
    let mut v: isize;
    v = v + 1; //~ ERROR E0381
    v.clone();
}

fn main() {
}
