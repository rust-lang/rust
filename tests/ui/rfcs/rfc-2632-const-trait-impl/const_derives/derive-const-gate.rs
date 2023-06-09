#[derive_const(Default)] //~ ERROR use of unstable library feature
//~^ ERROR not marked with `#[const_trait]`
pub struct S;

fn main() {}
