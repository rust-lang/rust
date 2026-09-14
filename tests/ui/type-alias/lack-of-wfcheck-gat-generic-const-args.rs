// Demonstrate that generic const arguments in GAT constraints are rejected at
// the definition site of an eager type alias.

//@ compile-flags: -Znext-solver=globally

#![feature(generic_const_args, min_generic_const_args)]
#![expect(incomplete_features)]

// * dyn incompatible due to GAT
// * `'a: 'static`, `String: Copy` and `[u8]: Sized` unsatisfied, `loop {}` diverging
type Several<'a> = dyn HasGenericAssocType<Type<'a, String, { loop {} }> = [u8]>;
//~^ ERROR

trait HasGenericAssocType {
    type Type<'a: 'static, T: Copy, const N: usize>;
}

fn main() {
    let _: &Several<'_>;
    //~^ ERROR the trait `HasGenericAssocType` is not dyn compatible
}
