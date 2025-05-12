extern crate bar;
extern crate foo;

pub struct Bar;
impl ::std::ops::Deref for Bar {
    type Target = bar::S;
    fn deref(&self) -> &Self::Target {
        unimplemented!()
    }
}
