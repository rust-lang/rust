mod a {
    enum Waffle {
        Belgian,
        Brussels,
        priv Liege
    }
}

fn main() {
    let x = a::Liege;   //~ ERROR unresolved name
}
