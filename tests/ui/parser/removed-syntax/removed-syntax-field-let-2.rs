struct Foo {
    let x: i32,
    //~^ ERROR expected identifier, found keyword
    let y: i32,
    //~^ ERROR expected identifier, found keyword
}

fn main() {
    let _ = Foo {
        //~^ ERROR missing fields `x` and `y` in initializer of `Foo`
    };
}
