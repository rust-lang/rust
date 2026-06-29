//@ aux-build:generics_other_crate.rs

extern crate generics_other_crate;
use generics_other_crate::External;

struct Local;
struct Defaulted<T = u32>(T);

// Case 1: Unused parameter -> suggests removing T
impl<T> Local {}
//~^ ERROR the type parameter `T` is not constrained by the impl trait, self type, or predicates
//~| HELP: remove the unused type parameter `T`

// Case 2: T used in body but not in self type
// -> suggests adding T to self type and struct definition
impl<T> Local {
    //~^ ERROR the type parameter `T` is not constrained by the impl trait, self type, or predicates
    //~| HELP: use the type parameter `T` in the `Local` type and use it in the type definition
    fn check() {
        let _: T;
    }
}


// Case 3: Struct has a generic parameter with a default
// -> suggests adding T to the self type
impl<T> Defaulted {}
//~^ ERROR the type parameter `T` is not constrained by the impl trait, self type, or predicates
//~| HELP: either remove the unused type parameter `T`
//~| HELP: or use it

// Case 4: Generated from a macro
macro_rules! make_impl {
    ($t:ident) => {
        impl<$t> Local {}
        //~^ HELP: remove the unused type parameter `U`
    }
}
make_impl!(U);
//~^ ERROR the type parameter `U` is not constrained by the impl trait, self type, or predicates

// Case 5: Type defined in another crate
impl<T> External {
    //~^ ERROR the type parameter `T` is not constrained by the impl trait, self type, or predicates
    //~| ERROR: cannot define inherent `impl` for a type outside of the crate where the type is defined [E0116]
    //~| HELP: use the type parameter `T` in the `External` type and use it in the type definition
    //~| HELP: consider defining a trait and implementing it for the type or using a newtype wrapper like `struct MyType(ExternalType);` and implement it
    fn check() {
        let _: T;
    }
}


fn main() {}
