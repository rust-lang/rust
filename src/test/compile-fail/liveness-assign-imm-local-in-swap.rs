fn test1() {
    let v: int;
    let mut w: int;
    v = 1; //! NOTE prior assignment occurs here
    w = 2;
    v <-> w; //! ERROR re-assignment of immutable variable
}

fn test2() {
    let v: int;
    let mut w: int;
    v = 1; //! NOTE prior assignment occurs here
    w = 2;
    w <-> v; //! ERROR re-assignment of immutable variable
}

fn main() {
}
