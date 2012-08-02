fn test() {
    let _v: int;
    _v = 1;
    return;
    _v = 2; //~ WARNING: unreachable statement
}

fn main() {
}
