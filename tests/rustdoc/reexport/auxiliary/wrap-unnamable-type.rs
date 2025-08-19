pub trait Assoc {
    type Ty;
}

pub struct Foo(<Foo as crate::Assoc>::Ty);

const _X: () = {
    impl crate::Assoc for Foo {
        type Ty = Bar;
    }
    pub struct Bar;
};
