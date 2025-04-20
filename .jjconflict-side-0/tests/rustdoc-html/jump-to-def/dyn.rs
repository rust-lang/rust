// This test ensures that `dyn` traits methods are correctly linked in the
// source code (and not to "foreigntype").

//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

//@ has 'src/foo/dyn.rs.html'
//@ has - '//a[@href="{{channel}}/core/any/trait.Any.html#method.is"]' 'is'
// There should be two of them.
//@ count - '//a[@href="{{channel}}/core/any/trait.Any.html#method.is"]' 2

pub struct X<T: ?Sized>(std::boxed::Box<T>);

impl X<dyn std::any::Any> {
    pub fn downcast<T: std::any::Any>(self) -> Result<X<T>, Self> {
        if self.0.is::<T>() {
            Ok(unsafe { std::mem::transmute(self) })
        } else {
            Err(self)
        }
    }
}

pub struct Y<T: ?Sized>(std::boxed::Box<T>);

impl Y<dyn std::any::Any + Send> {
    pub fn downcast<T: std::any::Any + Send>(self) -> Result<Y<T>, Self> {
        if self.0.is::<T>() {
            Ok(unsafe { std::mem::transmute(self) })
        } else {
            Err(self)
        }
    }
}
