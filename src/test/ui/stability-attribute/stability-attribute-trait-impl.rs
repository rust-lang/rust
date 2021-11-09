#![feature(staged_api)]
//~^ ERROR module has missing stability attribute

#[stable(feature = "a", since = "1")]
struct StableType;

#[unstable(feature = "b", issue = "none")]
struct UnstableType;

#[stable(feature = "c", since = "1")]
trait StableTrait {}

#[unstable(feature = "d", issue = "none")]
trait UnstableTrait {}

#[unstable(feature = "e", issue = "none")]
impl UnstableTrait for UnstableType {}

#[unstable(feature = "f", issue = "none")]
impl StableTrait for UnstableType {}

#[unstable(feature = "g", issue = "none")]
impl UnstableTrait for StableType {}

#[unstable(feature = "h", issue = "none")]
//~^ ERROR an `#[unstable]` annotation here has no effect [ineffective_unstable_trait_impl]
impl StableTrait for StableType {}

fn main() {}
