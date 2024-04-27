trait Lam {}

pub struct B;
impl Lam for B {}
pub struct Wrap<T>(T);

const _A: impl Lam = {
    //~^ `impl Trait` is not allowed in const types
    let x: Wrap<impl Lam> = Wrap(B);
    //~^ `impl Trait` is not allowed in the type of variable bindings
    x.0
};

fn main() {}
