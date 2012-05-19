fn test(cond: bool) {
    let mut v: int;
    v = v + 1; //! ERROR use of possibly uninitialized variable: `v`
}

fn main() {
    test(true);
}
