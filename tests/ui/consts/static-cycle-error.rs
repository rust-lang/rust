struct Foo {
    foo: Option<&'static Foo>
}

static FOO: Foo = Foo {
    foo: Some(&FOO),
    //~^ ERROR: encountered static that tried to access itself during initialization
};

fn main() {}
