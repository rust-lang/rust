macro_rules! m {
    ($p: path) => (pub(in $p) struct Z;)
}

struct S<T>(T);
m!{ S<u8> } //~ ERROR unexpected generic arguments in path
            //~| ERROR failed to resolve: `S` is a struct, not a module [E0433]

mod m {
    m!{ m<> } //~ ERROR unexpected generic arguments in path
}

fn main() {}
