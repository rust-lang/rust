macro_rules! m {
    ($p: path) => (pub(in $p) struct Z;)
}

struct S<T>(T);
m!{ crate::S<u8> } //~ ERROR unexpected generic arguments in path
                   //~| ERROR cannot find

mod m {
    m!{ crate::m<> } //~ ERROR unexpected generic arguments in path
}

fn main() {}
