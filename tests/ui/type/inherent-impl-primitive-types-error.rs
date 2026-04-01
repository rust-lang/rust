//! Test that inherent impl blocks cannot be defined for primitive types

impl u8 {
    //~^ ERROR: cannot define inherent `impl` for primitive types
    pub const B: u8 = 0;
}

impl str {
    //~^ ERROR: cannot define inherent `impl` for primitive types
    fn foo() {}
    fn bar(self) {} //~ ERROR: size for values of type `str` cannot be known
}

impl char {
    //~^ ERROR: cannot define inherent `impl` for primitive types
    pub const B: u8 = 0;
    pub const C: u8 = 0;
    fn foo() {}
    fn bar(self) {}
}

struct MyType;
impl &MyType {
    //~^ ERROR: cannot define inherent `impl` for primitive types
    pub fn for_ref(self) {}
}

fn main() {}
