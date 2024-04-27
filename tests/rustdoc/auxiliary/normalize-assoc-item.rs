#![crate_name = "inner"]
pub trait MyTrait {
    type Y;
}

impl MyTrait for u32 {
    type Y = i32;
}

pub fn foo() -> <u32 as MyTrait>::Y {
    0
}
