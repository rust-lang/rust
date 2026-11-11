fn test() {
    let v: isize;
    v += 1; //~ ERROR E0381
    v.clone();
}

fn main() {
}
