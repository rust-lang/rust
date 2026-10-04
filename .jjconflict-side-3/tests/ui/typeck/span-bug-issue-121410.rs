fn test_missing_unsafe_warning_on_repr_packed() {
    struct Foo {
        x: String,
    }

    let foo = Foo { x: String::new() };

    let c = || {
        let (_, t2) = foo.x; //~ ERROR mismatched types
    };

    c();
}

fn main() {}
