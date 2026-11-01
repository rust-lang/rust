// issue #41834

#[derive(Default)]
struct Foo {
    one: u8,
}

fn main() {
    let foo = Foo {
        one: 111,
        ..Foo::default(),
        //~^ ERROR cannot use a comma after the base struct
    };

    let foo = Foo {
        ..Foo::default(),
        //~^ ERROR cannot use a comma after the base struct
        one: 111,
    };
}
