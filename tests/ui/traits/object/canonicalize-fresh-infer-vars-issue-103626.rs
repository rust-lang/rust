trait FromResidual<R = <Self as Try>::Residual> {
    fn from_residual(residual: R) -> Self;
}

trait Try {
    type Residual;
}

fn w<'a, T: 'a, F: Fn(&'a T)>() {
    let b: &dyn FromResidual = &();
    //~^ ERROR: the trait `FromResidual` is not dyn compatible
    //~| ERROR the type parameter `R` must be explicitly specified
}

fn main() {}
