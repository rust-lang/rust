pub trait TraitWAssocConst {
    const A:   usize;
}
pub struct Demo {}

impl TraitWAssocConst for impl Demo { //~ ERROR E0404
    //~^ ERROR E0562
    pubconst A: str = 32; //~ ERROR expected one of
}

fn foo<A: TraitWAssocConst<A=32>>() { //~ ERROR E0658
    foo::<Demo>()();
    //~^ ERROR is not satisfied
    //~| ERROR expected function, found `()`
}

fn main<A: TraitWAssocConst<A=32>>() {
    //~^ ERROR E0658
    //~| ERROR E0131
    foo::<Demo>();
    //~^ ERROR is not satisfied
}
