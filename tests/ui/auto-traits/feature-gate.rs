// Test that auto traits are gated by the `rustc_attrs` feature gate.

#[rustc_auto_trait]
//~^ ERROR `#[rustc_auto_trait]` is used to mark auto traits, only intended to be used in `core`
trait AutoDummyTrait {}

fn main() {}
