pub struct Foo {
    pub foo: extern "C" fn()
}

extern "C" fn the_foo() {}

pub const FOO: Foo = Foo {
    foo: the_foo
};
