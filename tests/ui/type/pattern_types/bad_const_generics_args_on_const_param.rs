#![feature(pattern_types, pattern_type_macro)]
#![allow(internal_features)]

type Pat<const START: u32, const END: u32> =
    std::pat::pattern_type!(u32 is START::<(), i32, 2>..=END::<_, Assoc = ()>);
//~^ ERROR type and const arguments are not allowed on const parameter `START`
//~| ERROR type arguments are not allowed on const parameter `END`
//~| ERROR associated item constraints are not allowed here

fn main() {}
