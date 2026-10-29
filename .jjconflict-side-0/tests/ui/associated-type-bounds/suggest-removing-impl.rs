trait Tr {
    type Assoc: impl Sized;
    //~^ ERROR expected a trait, found type
    //~| HELP use the trait bounds directly

    fn fn_with_generics<T>()
    where
        T: impl Sized
        //~^ ERROR expected a trait, found type
        //~| HELP use the trait bounds directly
    {}
}

fn main() {}
