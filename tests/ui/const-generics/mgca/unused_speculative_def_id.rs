#![feature(min_generic_const_args, generic_const_exprs, generic_const_items)]
#![expect(incomplete_features)]

// Previously we would create a `DefId` to represent the const argument to `A`
// except it would go unused as it's a MGCA path const arg. We would also make
// a `DefId` for the `const { 1 }` anon const arg to `ERR` which would wind up
// with a `DefId` parent of the speculatively created `DefId` for the argument to
// `A`.
//
// This then caused Problems:tm: in the rest of the compiler that did not expect
// to encounter such nonsensical `DefId`s.
//
// The `ERR` path must fail to resolve as if it can be resolved then broken GCE
// logic will attempt to evaluate the constant directly which is wrong for
// `type_const`s which do not have bodies.

struct A<const N: usize>;

struct Foo {
    field: A<{ ERR::<const { 1 }> }>,
    //~^ ERROR: cannot find value `ERR` in this scope
}

fn main() {}
