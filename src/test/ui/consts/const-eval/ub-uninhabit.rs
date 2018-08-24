union Foo {
    a: u8,
    b: Bar,
}

#[derive(Copy, Clone)]
enum Bar {}

const BAD_BAD_BAD: Bar = unsafe { Foo { a: 1 }.b};
//~^ ERROR this constant likely exhibits undefined behavior

fn main() {
}
