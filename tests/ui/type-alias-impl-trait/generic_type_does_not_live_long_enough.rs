#![feature(type_alias_impl_trait)]

fn main() {
    let y = 42;
    let x = wrong_generic(&y);
    let z: i32 = x;
    //~^ ERROR expected generic type parameter, found `&i32`

    type WrongGeneric<T> = impl 'static;
    //~^ ERROR: at least one trait must be specified

    #[define_opaque(WrongGeneric)]
    fn wrong_generic<T>(t: T) -> WrongGeneric<T> {
        t
        //~^ ERROR the parameter type `T` may not live long enough
    }
}
