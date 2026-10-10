enum Foo<'a> {
    Bar(&'a u32)
}

fn in_let() {
    let y = 22;
    let foo = Foo::Bar(&y);
    //~^ ERROR `y` does not live long enough
    let Foo::Bar::<'static>(_z) = foo;
}

fn in_match() {
    let y = 22;
    let foo = Foo::Bar(&y);
    //~^ ERROR `y` does not live long enough
    match foo {
        Foo::Bar::<'static>(_z) => {
        }
    }
}

fn main() { }
