fn test(cond: bool) {
    let v: int;
    v = 2;  //! NOTE prior assignment occurs here
    v += 1; //! ERROR re-assignment of immutable variable
}

fn main() {
    test(true);
}
