trait Lam {}

pub struct B;
impl Lam for B {}
pub struct Wrap<T>(T);

const _A: impl Lam = {
    //~^ ERROR `impl Trait` is not allowed in const types
    let x: Wrap<impl Lam> = Wrap(B);
    //~^ ERROR `impl Trait` is not allowed in the type of variable bindings
    x.0
};

fn main() {}
