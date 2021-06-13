pub mod module {
    pub mod sub_module {
        pub mod sub_sub_module {
            pub fn foo() {}
        }
        pub fn bar() {}
    }
    pub fn whatever() {}
}

pub fn foobar() {}

pub type Alias = u32;

pub struct Foo {
    pub x: Alias,
}

impl Foo {
    pub fn a_method(&self) {}
}

pub trait Trait {
    type X;
    const Y: u32;
}

impl Trait for Foo {
    type X = u32;
    const Y: u32 = 0;
}
