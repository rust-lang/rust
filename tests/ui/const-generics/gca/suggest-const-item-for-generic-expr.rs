// Regression test for https://github.com/rust-lang/rust/issues/156729
//
// When a generic parameter is used in a const operation, the diagnostic should
// suggest creating a `type const` item as an alternative to `generic_const_exprs`.

use std::mem::size_of;

// Exact case from the issue: `Self` in a trait method.
pub unsafe trait TrivialType: Copy {
    fn as_bytes(&self) -> &[u8; size_of::<Self>()] {
        //~^ ERROR generic parameters may not be used in const operations
        todo!()
    }
}

fn foo<T>() -> [u8; size_of::<T>()] {
    //~^ ERROR generic parameters may not be used in const operations
    todo!()
}

fn bar<const N: usize>() -> [u8; N + 1] {
    //~^ ERROR generic parameters may not be used in const operations
    todo!()
}

fn main() {}
