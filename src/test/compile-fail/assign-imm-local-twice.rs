fn test() {
    let v: int;
    v = 1; //! NOTE prior assignment occurs here
    #debug["v=%d", v];
    v = 2; //! ERROR re-assignment of immutable variable
    #debug["v=%d", v];
}

fn main() {
}
