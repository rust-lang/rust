struct Foo {
    bar: u8
}

fn main() {
    let f = Foo { bar: 22 };
    f.zz; //~ ERROR no field
}
