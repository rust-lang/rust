#[derive_const(Default)] //~ ERROR use of unstable library feature
//~^ ERROR const `impl` for trait `Default` which is not marked with `#[const_trait]`
pub struct S;

fn main() {}
