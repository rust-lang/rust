fn test(cond: bool) {
    let v;
    while cond {
        v = 3;
        break;
    }
    println!("{}", v); //~ ERROR E0381
}

fn main() {
    test(true);
}
