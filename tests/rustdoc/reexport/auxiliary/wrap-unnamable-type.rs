pub trait Assoc {
    type Ty;
}

pub struct Foo(<Foo as crate::Assoc>::Ty);

const _: () = {
    impl crate::Assoc for Foo {
        type Ty = Bar;
    }
    pub struct Bar;
};
