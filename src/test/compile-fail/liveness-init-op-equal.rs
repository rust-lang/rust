fn test() {
    let v: int;
    v += 1; //~ ERROR use of possibly uninitialized variable: `v`
    copy v;
}

fn main() {
}
