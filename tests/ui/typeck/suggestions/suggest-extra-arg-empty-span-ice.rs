// Regression test for https://github.com/rust-lang/rust/issues/152414

#![feature(generic_assert)]
pub const fn make_1u8_bag<T: Copy>() -> BagOfBits<_> {
    //~^ ERROR cannot find type `BagOfBits` in this scope
    //~| ERROR the placeholder `_` is not allowed within types on item signatures for return types
    assert!(core::mem::size_of::<T>(val, 1) >= 1);
    //~^ ERROR cannot find value `val` in this scope
    //~| ERROR this function takes 0 arguments but 2 arguments were supplied
    bag
    //~^ ERROR cannot find value `bag` in this scope
}

fn main() {}
