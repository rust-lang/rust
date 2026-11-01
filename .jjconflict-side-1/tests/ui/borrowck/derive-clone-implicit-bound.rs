// Issue #108894

use std::marker::PhantomData;

#[derive(Clone, Copy)] //~ NOTE: derived `Clone` adds implicit bounds on type parameters
pub struct TypedAddress<T>{
//~^ NOTE: if all bounds were met, you could clone the value
//~| NOTE: introduces an implicit `T: Clone` bound
    inner: u64,
    phantom: PhantomData<T>,
}

pub trait Memory {
    fn write_value<T>(&self, offset: TypedAddress<T>, value: &T);
    fn return_value<T>(&self, offset: TypedAddress<T>) -> T;
    //~^ NOTE: consider changing this parameter type in method `return_value` to borrow instead if owning the value isn't necessary
    //~| NOTE: in this method
    //~| NOTE: this parameter takes ownership of the value
    fn update_value<T, F>(&self, offset: TypedAddress<T>, update: F)
    //~^ NOTE: move occurs because `offset` has type `TypedAddress<T>`, which does not implement the `Copy` trait
        where F: FnOnce(T) -> T //~ HELP: consider further restricting type parameter `T`
    {
        let old = self.return_value(offset); //~ NOTE: value moved here
        //~^ NOTE: you could clone this value
        let new = update(old);
        self.write_value(offset, &new); //~ ERROR: use of moved value: `offset`
        //~^ NOTE: value used here after move
        //~| HELP: consider manually implementing `Clone` to avoid undesired bounds
    }
}

fn main() {}
