//~ ERROR the parameter type `Self` may not live long enough

trait GatTrait {
    type Gat<'a>
    where
        Self: 'a;
}

trait SuperTrait<T>: for<'a> GatTrait<Gat<'a> = T> {
    fn c(&self) -> dyn SuperTrait<T>;
    //~^ ERROR associated item referring to unboxed trait object for its own trait
    //~| ERROR the trait `SuperTrait` is not dyn compatible
}

fn main() {}
