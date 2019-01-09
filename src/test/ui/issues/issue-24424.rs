trait Trait1<'l0, T0> {}
trait Trait0<'l0>  {}

impl <'l0, 'l1, T0> Trait1<'l0, T0> for bool where T0 : Trait0<'l0>, T0 : Trait0<'l1> {}
//~^ ERROR type annotations required: cannot resolve `T0: Trait0<'l0>`

fn main() {}
