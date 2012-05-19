fn test(cond: bool) {
    let v: int;
    v = 1; //! NOTE prior assignment occurs here
    v = 2; //! ERROR re-assignment of immutable variable
}

fn main() {
    test(true);
}
