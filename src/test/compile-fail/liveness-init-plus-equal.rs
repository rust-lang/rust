fn test() {
    let mut v: int;
    v = v + 1; //! ERROR use of possibly uninitialized variable: `v`
    copy v;
}

fn main() {
}
