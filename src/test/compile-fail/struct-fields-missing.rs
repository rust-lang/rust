struct BuildData {
    foo: int,
    bar: ~int
}

fn main() {
    let foo = BuildData { //~ ERROR missing field: `bar`
        foo: 0
    };
}
