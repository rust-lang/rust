impl u8 {
//~^ error: cannot define inherent `impl` for primitive types
    pub const B: u8 = 0;
}

impl str {
//~^ error: cannot define inherent `impl` for a type outside of the crate where the type is defined
    fn foo() {}
    fn bar(self) {}
}

impl char {
//~^ error: cannot define inherent `impl` for primitive types
    pub const B: u8 = 0;
    pub const C: u8 = 0;
    fn foo() {}
    fn bar(self) {}
}

struct MyType;
impl &MyType {
//~^ error: cannot define inherent `impl` for primitive types
    pub fn for_ref(self) {}
}

fn main() {}
