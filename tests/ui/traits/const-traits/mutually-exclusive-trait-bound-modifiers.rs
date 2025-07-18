#![feature(const_trait_impl)]

const fn maybe_const_maybe<T: [const] ?Sized>() {}
//~^ ERROR `[const]` trait not allowed with `?` trait polarity modifier
//~| ERROR `[const]` can only be applied to `const` traits
//~| ERROR `[const]` can only be applied to `const` traits

fn const_maybe<T: const ?Sized>() {}
//~^ ERROR `const` trait not allowed with `?` trait polarity modifier
//~| ERROR `const` can only be applied to `const` traits

const fn maybe_const_negative<T: [const] !Trait>() {}
//~^ ERROR `[const]` trait not allowed with `!` trait polarity modifier
//~| ERROR negative bounds are not supported

fn const_negative<T: const !Trait>() {}
//~^ ERROR `const` trait not allowed with `!` trait polarity modifier
//~| ERROR negative bounds are not supported

#[const_trait]
trait Trait {}

fn main() {}
