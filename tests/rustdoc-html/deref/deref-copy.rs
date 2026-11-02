// This test ensures that when we `Deref` to a type implementing `Copy`, the
// methods with `self` are also kept.

#![crate_name = "foo"]

#[derive(Clone, Copy)]
pub struct Y;

impl Y {
    pub const ONE: X = X { y: Y };

    pub fn foo() {}
    pub fn bar(&self) {}
    pub fn babar(self) {}
}

// There should be `bar` and `babar` listed... and that's it!
//@ has 'foo/struct.X.html'
//@ count - '//*[@id="deref-methods-Y-1"]/section' 2
//@ has - '//*[@id="deref-methods-Y-1"]/section[@id="method.bar"]' 'pub fn bar(&self)'
//@ has - '//*[@id="deref-methods-Y-1"]/section[@id="method.babar"]' 'pub fn babar(self)'
pub struct X {
    y: Y,
}

impl std::ops::Deref for X {
    type Target = Y;

    fn deref(&self) -> &Self::Target {
        &self.y
    }
}
