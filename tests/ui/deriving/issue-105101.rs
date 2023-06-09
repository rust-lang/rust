// compile-flags: --crate-type=lib

#[derive(Default)] //~ ERROR multiple declared defaults
enum E {
    #[default]
    A,
    #[default]
    A, //~ ERROR defined multiple times
}
