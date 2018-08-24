struct BuildData {
    foo: isize,
    bar: f32
}

fn main() {
    let foo = BuildData {
        foo: 0,
        bar: 0.5,
    };
    let x = foo.baa;//~ no field `baa` on type `BuildData`
    //~^ did you mean `bar`?
    println!("{}", x);
}
