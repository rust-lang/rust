#![feature(type_alias_impl_trait)]

mod lifetime_params {
    type Ty<'a> = impl Sized;
    fn defining(s: &str) -> Ty<'_> { s }
    fn execute(ty: Ty<'_>) -> &str { todo!() }
    //~^ ERROR return type references an anonymous lifetime, which is not constrained by the fn input types

    type BadFnSig = fn(Ty<'_>) -> &str;
    //~^ ERROR return type references an anonymous lifetime, which is not constrained by the fn input types
    type BadTraitRef = dyn Fn(Ty<'_>) -> &str;
    //~^ ERROR binding for associated type `Output` references an anonymous lifetime
}

mod lifetime_params_2 {
    type Ty<'a> = impl FnOnce() -> &'a str;
    fn defining(s: &str) -> Ty<'_> { move || s }
    fn execute(ty: Ty<'_>) -> &str { ty() }
    //~^ ERROR return type references an anonymous lifetime, which is not constrained by the fn input types
}

// regression test for https://github.com/rust-lang/rust/issues/97104
mod type_params {
    type Ty<T> = impl Sized;
    fn define<T>(s: T) -> Ty<T> { s }

    type BadFnSig = fn(Ty<&str>) -> &str;
    //~^ ERROR return type references an anonymous lifetime, which is not constrained by the fn input types
    type BadTraitRef = dyn Fn(Ty<&str>) -> &str;
    //~^ ERROR binding for associated type `Output` references an anonymous lifetime
}

fn main() {}
