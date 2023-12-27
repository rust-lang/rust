// Regression test for issue #117244.
#![feature(const_trait_impl, effects)]

trait NonConst {}

const fn perform<T: ~const NonConst>() {}
//~^ ERROR `~const` can only be applied to `#[const_trait]` traits

fn operate<T: const NonConst>() {}
//~^ ERROR `const` can only be applied to `#[const_trait]` traits

fn main() {}
