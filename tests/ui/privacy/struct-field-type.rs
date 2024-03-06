mod m {
    struct Priv;
    pub type Leak = Priv; //~ WARN: `Priv` is more private than the item `Leak`
}

struct S {
    field: m::Leak, //~ ERROR: `Priv` is private
}

fn main() {}
