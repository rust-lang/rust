mod a {
    #[legacy_exports];
    struct Foo {
        priv x: int
    }
}

fn main() {
    let s = a::Foo { x: 1 };    //~ ERROR field `x` is private
}

