enum E {
    A,
    B,
    C,
    D,
    E,
    F,
}

fn main() {
    match E::F {
        E::A => 1,
        E::B => 2,
        E::C => 3,
        E::D => 4,
        E::E => unimplemented!(""),
        E::F => "", //~ ERROR `match` arms have incompatible types
    };
}
