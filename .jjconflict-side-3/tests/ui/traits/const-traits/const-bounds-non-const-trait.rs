// Regression test for issue #117244.
#![feature(const_trait_impl)]

trait NonConst {}

const fn perform<T: [const] NonConst>() {}
//~^ ERROR `[const]` can only be applied to `const` traits
//~| ERROR `[const]` can only be applied to `const` traits

fn operate<T: const NonConst>() {}
//~^ ERROR `const` can only be applied to `const` traits

fn main() {}
