//@compile-flags: -Znext-solver=globally --crate-type=lib
pub trait Trait {
    type Assoc;
}

pub fn foo<T: Trait<Assoc = u32> + Trait<Assoc = i32>>() {
    //~^ ERROR type annotations needed: cannot satisfy `<T as Trait>::Assoc == u32` [E0284]
    const {}
}
