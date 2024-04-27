// regression test for #74018

#![feature(impl_trait_in_assoc_type)]

trait Trait {
    type Associated;
    fn into(self) -> Self::Associated;
}

impl<'a, I: Iterator<Item = i32>> Trait for (i32, I) {
    //~^ ERROR the lifetime parameter `'a` is not constrained
    type Associated = (i32, impl Iterator<Item = i32>);
    fn into(self) -> Self::Associated {
        (0_i32, [0_i32].iter().copied())
        //~^ ERROR: expected generic lifetime parameter, found `'_`
    }
}

fn main() {}
