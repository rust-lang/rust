// check-fail

#![deny(where_clauses_object_safety)]

unsafe trait UnsafeTrait<T: ?Sized> {}

trait Trait {
    fn unsafe_trait_bound(&self) where (): UnsafeTrait<Self> {} //~ ERROR: the trait `Trait` cannot be made into an object
    //~^ WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn main() {
    let _: &dyn Trait;
}
