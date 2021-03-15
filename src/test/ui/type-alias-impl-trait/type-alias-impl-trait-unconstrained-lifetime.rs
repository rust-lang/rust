// regression test for #74018

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

trait Trait {
    type Associated;
    fn into(self) -> Self::Associated;
}

impl<'a, I: Iterator<Item = i32>> Trait for (i32, I) {
    //~^ ERROR the lifetime parameter `'a` is not constrained
    type Associated = (i32, impl Iterator<Item = i32>);
    fn into(self) -> Self::Associated {
        (0_i32, [0_i32].iter().copied())
    }
}

fn main() {}
