#![feature(const_trait_impl)]

const fn tilde_question<T: ~const ?Sized>() {}
//~^ ERROR `~const` and `?` are mutually exclusive

fn main() {}
