#![feature(type_alias_impl_trait)]

mod lifetime_params {
    type Ty<'a> = impl Sized;
    #[defines(Ty)]
    fn defining(s: &str) -> Ty<'_> { s }
    #[defines(Ty)]
    fn execute(ty: Ty<'_>) -> &str { todo!() }
    //~^ ERROR return type references an anonymous lifetime, which is not constrained by the fn input types
    //~| ERROR item does not constrain

    type BadFnSig = fn(Ty<'_>) -> &str;
    //~^ ERROR return type references an anonymous lifetime, which is not constrained by the fn input types
    type BadTraitRef = dyn Fn(Ty<'_>) -> &str;
    //~^ ERROR binding for associated type `Output` references an anonymous lifetime
}

mod lifetime_params_2 {
    type Ty<'a> = impl FnOnce() -> &'a str;
    #[defines(Ty)]
    fn defining(s: &str) -> Ty<'_> { move || s }
    #[defines(Ty)]
    fn execute(ty: Ty<'_>) -> &str { ty() }
    //~^ ERROR return type references an anonymous lifetime, which is not constrained by the fn input types
    //~| ERROR item does not constrain
}

// regression test for https://github.com/rust-lang/rust/issues/97104
mod type_params {
    type Ty<T> = impl Sized;
    #[defines(Ty)]
    fn define<T>(s: T) -> Ty<T> { s }

    type BadFnSig = fn(Ty<&str>) -> &str;
    //~^ ERROR return type references an anonymous lifetime, which is not constrained by the fn input types
    type BadTraitRef = dyn Fn(Ty<&str>) -> &str;
    //~^ ERROR binding for associated type `Output` references an anonymous lifetime
}

fn main() {}
