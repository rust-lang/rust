trait Arr0 {
    fn arr0_secret(&self);
}
trait TyParam {
    fn ty_param_secret(&self);
}

trait Ref {
    fn ref_secret(self);
}

mod m {
    struct Priv;

    impl crate::Arr0 for [Priv; 0] { fn arr0_secret(&self) {} }
    impl crate::TyParam for Option<Priv> { fn ty_param_secret(&self) {} }
    impl<'a> crate::Ref for &'a Priv { fn ref_secret(self) {} }
}

fn anyref<'a, T>() -> &'a T { panic!() }

fn main() {
    [].arr0_secret(); //~ ERROR type `Priv` is private
    None.ty_param_secret(); //~ ERROR type `Priv` is private
    Ref::ref_secret(anyref());
    //~^ ERROR type `Priv` is private
    //~| ERROR type `Priv` is private
}
