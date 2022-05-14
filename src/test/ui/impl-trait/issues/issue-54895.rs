trait Trait<'a> {
    type Out;
    fn call(&'a self) -> Self::Out;
}

struct X(());

impl<'a> Trait<'a> for X {
    type Out = ();
    fn call(&'a self) -> Self::Out {
        ()
    }
}

fn f() -> impl for<'a> Trait<'a, Out = impl Sized + 'a> {
    //~^ ERROR higher kinded lifetime bounds on nested opaque types are not supported yet
    X(())
}

fn main() {
    let _ = f();
}
