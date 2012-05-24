fn test() {
    let v: int;
    v = 2;  //! NOTE prior assignment occurs here
    v += 1; //! ERROR re-assignment of immutable variable
    copy v;
}

fn main() {
}
