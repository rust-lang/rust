fn test(cond: bool) {
    let v: int;
    v += 1; //! ERROR use of possibly uninitialized variable: `v`
}

fn main() {
    test(true);
}
