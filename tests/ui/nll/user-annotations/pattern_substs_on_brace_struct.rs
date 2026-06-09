struct Foo<'a> { field: &'a u32 }

fn in_let() {
    let y = 22;
    let foo = Foo { field: &y };
    //~^ ERROR `y` does not live long enough
    let Foo::<'static> { field: _z } = foo;
}

fn in_main() {
    let y = 22;
    let foo = Foo { field: &y };
    //~^ ERROR `y` does not live long enough
    match foo {
        Foo::<'static> { field: _z } => {
        }
    }
}

fn main() { }
