trait Lam {}

pub struct B;
impl Lam for B {}
pub struct Wrap<T>(T);

const _A: impl Lam = {
    //~^ `impl Trait` not allowed outside of function and inherent method return types
    let x: Wrap<impl Lam> = Wrap(B);
    //~^ `impl Trait` not allowed outside of function and inherent method return types
    x.0
};

fn main() {}
