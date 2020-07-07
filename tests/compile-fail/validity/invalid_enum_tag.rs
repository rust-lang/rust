#[repr(C)]
pub enum Foo {
    A, B, C, D
}

fn main() {
    let _f = unsafe { std::mem::transmute::<i32, Foo>(42) }; //~ ERROR encountered 0x0000002a at .<enum-tag>, but expected a valid enum tag
}
