struct BuildData {
    foo: isize,
    bar: f32
}

fn main() {
    let foo = BuildData {
        foo: 0,
        bar: 0.5,
    };
    let x = foo.baa; //~ ERROR no field `baa` on type `BuildData`
                     //~| HELP a field with a similar name exists
                     //~| SUGGESTION bar
    println!("{}", x);
}
