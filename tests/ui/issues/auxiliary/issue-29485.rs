#![crate_name="a"]
#![crate_type = "lib"]

pub struct X(pub u8);

impl Drop for X {
    fn drop(&mut self) {
        assert_eq!(self.0, 1)
    }
}

pub fn f(x: &mut X, g: fn()) {
    x.0 = 1;
    g();
    x.0 = 0;
}
