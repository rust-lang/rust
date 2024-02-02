fn make<N: u32>() {}
//~^ ERROR expected trait, found builtin type `u32`
//~| HELP you might have meant to write a const parameter here

struct Array<N: usize>([bool; N]);
//~^ ERROR expected trait, found builtin type `usize`
//~| HELP you might have meant to write a const parameter here
//~| ERROR expected value, found type parameter `N`

fn main() {}
