macro_rules! m {
    ($p: path) => (pub(in $p) struct Z;)
}

struct S<T>(T);
m!{ S<u8> } //~ ERROR unexpected generic arguments in path
            //~| ERROR expected module, found struct `S`

mod m {
    m!{ m<> } //~ ERROR unexpected generic arguments in path
}

fn main() {}
