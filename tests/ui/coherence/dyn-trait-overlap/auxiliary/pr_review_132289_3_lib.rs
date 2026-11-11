use std::ops::Index;

pub trait Trait {
    fn f(&self)
    where
        dyn Index<(), Output = ()>: Index<()>;
    //  rustc (correctly) determines ^^^^^^^^ this bound to be true
}

pub fn call(x: &dyn Trait) {
    x.f(); // so we can call `f`
}
