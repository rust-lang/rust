#[repr(align(16))]
//~^ ERROR attribute cannot be used on
pub type Foo = i32;

fn main() {}
