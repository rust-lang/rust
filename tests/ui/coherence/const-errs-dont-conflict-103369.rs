// #103369: don't complain about conflicting implementations with [const error]

pub trait ConstGenericTrait<const N: u32> {}

impl ConstGenericTrait<{my_fn(1)}> for () {}

impl ConstGenericTrait<{my_fn(2)}> for () {}

const fn my_fn(v: u32) -> u32 {
    panic!("Some error occurred"); //~ ERROR E0080
    //~| ERROR E0080
}

fn main() {}
