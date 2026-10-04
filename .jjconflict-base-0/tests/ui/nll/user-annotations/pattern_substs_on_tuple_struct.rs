struct Foo<'a>(&'a u32);

fn in_let() {
    let y = 22;
    let foo = Foo(&y);
    //~^ ERROR `y` does not live long enough
    let Foo::<'static>(_z) = foo;
}

fn in_match() {
    let y = 22;
    let foo = Foo(&y);
    //~^ ERROR `y` does not live long enough
    match foo {
        Foo::<'static>(_z) => {
        }
    }
}

fn main() { }
