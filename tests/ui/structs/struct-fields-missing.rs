struct BuildData {
    foo: isize,
    bar: Box<isize>,
}

fn main() {
    let foo = BuildData { //~ ERROR missing field `bar` in initializer of `BuildData`
        foo: 0
    };
}
