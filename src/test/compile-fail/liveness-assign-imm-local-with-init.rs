fn test() {
    let v: int = 1; //! NOTE prior assignment occurs here
    copy v;
    v = 2; //! ERROR re-assignment of immutable variable
    copy v;
}

fn main() {
}
