trait Lam {}

pub struct B;
impl Lam for B {}
pub struct Wrap<T>(T);

const _A: impl Lam = {
    //~^ `impl Trait` not allowed within type
    let x: Wrap<impl Lam> = Wrap(B);
    //~^ `impl Trait` not allowed within variable binding
    x.0
};

fn main() {}
