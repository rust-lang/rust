fn test(cond: bool) {
    let v: int;
    loop {
        v = 1; //! ERROR re-assignment of immutable variable
        //!^ NOTE prior assignment occurs here
    }
}

fn main() {
    test(true);
}
