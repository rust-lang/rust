mod a {
    #[legacy_exports];
    priv fn f() {}
}

fn main() {
    a::f(); //~ ERROR unresolved name
}

