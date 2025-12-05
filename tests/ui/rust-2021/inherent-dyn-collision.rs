// Test case where the method we want is an inherent method on a
// dyn Trait. In that case, the fix is to insert `*` on the receiver.
//
//@ check-pass
//@ run-rustfix
//@ edition:2018

#![warn(rust_2021_prelude_collisions)]

trait TryIntoU32 {
    fn try_into(&self) -> Result<u32, ()>;
}

impl TryIntoU32 for u8 {
    // note: &self
    fn try_into(&self) -> Result<u32, ()> {
        Ok(22)
    }
}

mod inner {
    use super::get_dyn_trait;

    // note: this does nothing, but is copying from ffishim's problem of
    // having a struct of the same name as the trait in-scope, while *also*
    // implementing the trait for that struct but **without** importing the
    // trait itself into scope
    #[allow(dead_code)]
    struct TryIntoU32;

    impl super::TryIntoU32 for TryIntoU32 {
        fn try_into(&self) -> Result<u32, ()> {
            Ok(0)
        }
    }

    // this is where the gross part happens. since `get_dyn_trait` returns
    // a Box<dyn Trait>, it can still call the method for `dyn Trait` without
    // `Trait` being in-scope. it might even be possible to make the trait itself
    // entirely unreference-able from the callsite?
    pub fn test() -> u32 {
        get_dyn_trait().try_into().unwrap()
        //~^ WARNING trait method `try_into` will become ambiguous
        //~| WARNING this is accepted in the current edition
    }
}

fn get_dyn_trait() -> Box<dyn TryIntoU32> {
    Box::new(3u8) as Box<dyn TryIntoU32>
}

fn main() {
    dbg!(inner::test());
}
