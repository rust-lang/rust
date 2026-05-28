enum OhNo<T, U> {
    A(T),
    B(U),
    C,
}

fn uwu() {
    OhNo::C::<u32, _>; //~ ERROR type annotations needed
}

fn main() {}
