#[repr(align(16))]
//~^ ERROR attribute should be applied to a struct, enum, or union
pub type Foo = i32;

fn main() {}
