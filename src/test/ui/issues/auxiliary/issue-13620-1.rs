pub struct Foo {
    pub foo: extern fn()
}

extern fn the_foo() {}

pub const FOO: Foo = Foo {
    foo: the_foo
};
