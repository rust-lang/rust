//@run-fail

#![feature(type_info)]

trait Trait {}

fn main() {
    // Test the (lack of) usability of comptime fns in runtime code.
    std::any::TypeId::of::<[u8; usize::MAX]>().trait_info_of::<dyn Trait>();
}
