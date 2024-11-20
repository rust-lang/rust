//@ edition:2018

#![feature(allocator_api)]
struct Struct;
impl Struct {
    async fn box_ref_Struct(self: Box<Self, impl FnMut(&mut Self)>) -> &u32 {
        //~^ ERROR the trait bound
        &1
    }
}

fn main() {}
