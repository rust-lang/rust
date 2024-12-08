pub trait TraitWAssocConst {
  const A: usize;
}
pub struct Demo {}

impl TraitWAssocConst for Demo {
  const A: usize = 32;
}

fn foo<A: TraitWAssocConst<A=32>>() {}
//~^ ERROR associated const equality

fn main() {
  foo::<Demo>();
}
