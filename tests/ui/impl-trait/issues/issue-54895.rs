//@revisions: edition2015 edition2024
//@[edition2015] edition:2015
//@[edition2024] edition:2024
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
    //~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from outer `impl Trait`
    //[edition2024]~^^ ERROR `impl Trait` cannot capture higher-ranked lifetime from outer `impl Trait`
    X(())
}

fn main() {
    let _ = f();
}
