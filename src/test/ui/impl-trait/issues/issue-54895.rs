// check-pass

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
    X(())
}

fn main() {
    let _ = f();
}
