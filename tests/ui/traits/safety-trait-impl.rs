// Check that unsafe traits require unsafe impls and that inherent
// impls cannot be unsafe.

trait SafeTrait {
    fn foo(&self) { }
}

unsafe trait UnsafeTrait {
    fn foo(&self) { }
}

unsafe impl UnsafeTrait for u8 { } // OK

impl UnsafeTrait for u16 { } //~ ERROR requires an `unsafe impl` declaration

unsafe impl SafeTrait for u32 { } //~ ERROR the trait `SafeTrait` is not unsafe

fn main() { }
