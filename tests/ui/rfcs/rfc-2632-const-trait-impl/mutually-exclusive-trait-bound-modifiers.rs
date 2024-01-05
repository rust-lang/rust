#![feature(const_trait_impl)]

const fn maybe_const_maybe<T: ~const ?Sized>() {}
//~^ ERROR `~const` and `?` are mutually exclusive

fn const_maybe<T: const ?Sized>() {}
//~^ ERROR `const` and `?` are mutually exclusive

const fn maybe_const_negative<T: ~const !Trait>() {}
//~^ ERROR `~const` and `!` are mutually exclusive
//~| ERROR negative bounds are not supported

fn const_negative<T: const !Trait>() {}
//~^ ERROR `const` and `!` are mutually exclusive
//~| ERROR negative bounds are not supported

#[const_trait]
trait Trait {}

fn main() {}
