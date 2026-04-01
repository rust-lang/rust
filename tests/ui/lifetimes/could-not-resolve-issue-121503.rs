//@ edition:2018

#![feature(allocator_api)]
struct Struct;
impl Struct {
    async fn box_ref_Struct(self: Box<Self, impl FnMut(&mut Self)>) -> &u32 {
    //~^ ERROR Box<Struct, impl FnMut(&mut Self)>` cannot be used as the type of `self` without
        &1
    }
}

fn main() {}
