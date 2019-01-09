struct BuildData {
    foo: isize,
}

fn main() {
    let foo = BuildData {
        foo: 0,
        foo: 0 //~ ERROR field `foo` specified more than once
    };
}
