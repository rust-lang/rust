trait Arr0 {
    fn arr0_secret(&self);
}
trait TyParam {
    fn ty_param_secret(&self);
}

mod m {
    struct Priv;

    impl ::Arr0 for [Priv; 0] { fn arr0_secret(&self) {} }
    impl ::TyParam for Option<Priv> { fn ty_param_secret(&self) {} }
}

fn main() {
    [].arr0_secret(); //~ ERROR type `m::Priv` is private
    None.ty_param_secret(); //~ ERROR type `m::Priv` is private
}
