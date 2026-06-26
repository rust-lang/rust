struct BuildData {
    foo: isize,
}

fn main() {
    let foo = BuildData {
        foo: 0,
        bar: 0
        //~^ ERROR struct `BuildData` has no field named `bar`
    };
}
