trait Foo {
    fn bar() {}
}

fn main() {
    Foo::bar();
    //~^ ERROR type annotations needed

    <_ as Foo>::bar();
}
