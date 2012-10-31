struct BuildData {
    foo: int,
}

fn main() {
    let foo = BuildData {
        foo: 0,
        bar: 0 //~ ERROR structure has no field named `bar`
    };
}
