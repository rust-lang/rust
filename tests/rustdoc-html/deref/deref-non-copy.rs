// This test ensures that when we `Deref` to a type doesn't implement `Copy`, the
// methods with `self` are not kept.

#![crate_name = "foo"]

pub struct Y;

impl Y {
    pub const ONE: X = X { y: Y };

    pub fn foo() {}
    pub fn bar(&self) {}
    pub fn babar(self) {}
}

// There should be `bar` listed... and that's it!
//@ has 'foo/struct.X.html'
//@ count - '//*[@id="deref-methods-Y-1"]/section' 1
//@ has - '//*[@id="deref-methods-Y-1"]/section[@id="method.bar"]' 'pub fn bar(&self)'
pub struct X {
    y: Y,
}

impl std::ops::Deref for X {
    type Target = Y;

    fn deref(&self) -> &Self::Target {
        &self.y
    }
}
