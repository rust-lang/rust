pub mod blah {
    pub mod baz {
        pub struct Foo;
    }
}

pub mod meh {
    pub struct Foo;
}

pub type Foo = blah::baz::Foo;

fn foo() -> Foo {
    meh::Foo
    //~^ ERROR mismatched types [E0308]
}

fn main() {}
