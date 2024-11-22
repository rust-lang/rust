trait FromResidual<R = <Self as Try>::Residual> {
    fn from_residual(residual: R) -> Self;
}

trait Try {
    type Residual;
}

fn w<'a, T: 'a, F: Fn(&'a T)>() {
    let b: &dyn FromResidual = &();
    //~^ ERROR: the trait `FromResidual` cannot be made into an object
    //~| ERROR: the trait `FromResidual` cannot be made into an object
    //~| ERROR: the trait `FromResidual` cannot be made into an object
}

fn main() {}
