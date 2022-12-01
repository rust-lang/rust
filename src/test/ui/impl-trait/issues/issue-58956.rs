trait Lam {}

pub struct B;
impl Lam for B {}
pub struct Wrap<T>(T);

const _A: impl Lam = {
    //~^ `impl Trait` isn't allowed within type
    let x: Wrap<impl Lam> = Wrap(B);
    //~^ `impl Trait` isn't allowed within variable binding
    x.0
};

fn main() {}
