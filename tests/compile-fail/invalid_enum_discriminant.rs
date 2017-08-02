// Validation makes this fail in the wrong place
// compile-flags: -Zmir-emit-validate=0

#[repr(C)]
pub enum Foo {
    A, B, C, D
}

fn main() {
    let f = unsafe { std::mem::transmute::<i32, Foo>(42) };
    match f {
        Foo::A => {}, //~ ERROR invalid enum discriminant value read
        Foo::B => {},
        Foo::C => {},
        Foo::D => {},
    }
}
