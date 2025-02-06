pub struct Data([[&'static str]; 5_i32]);
//~^ ERROR: mismatched types
//~| ERROR: the size for values of type `[&'static str]`
const _: &'static Data = unsafe { &*(&[] as *const Data) };

fn main() {}
