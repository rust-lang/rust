mod a {
    priv fn f() {}
}

fn main() {
    a::f(); //~ ERROR unresolved name
}

