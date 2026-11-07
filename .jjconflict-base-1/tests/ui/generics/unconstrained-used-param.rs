//! Test that making use of parameter is suggested when the parameter is used in the impl
//! but not in the trait or self type

struct S;
impl<T> S {
//~^ ERROR the type parameter `T` is not constrained by the impl trait, self type, or predicates
   fn foo(&self, x: T) {
       // use T here
   }
}


struct S2<F> {
    _f: F,
}
impl<F, T> S2<F> {
//~^ ERROR the type parameter `T` is not constrained by the impl trait, self type, or predicates
   fn foo(&self, x: T) {
       // use T here
   }
}

fn main() {}
