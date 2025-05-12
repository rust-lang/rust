struct Foo {
    x: i32
}

fn main() {
    let x = Foo {
        x: 0,
        x: 0,
        //~^ ERROR E0062
    };
}
