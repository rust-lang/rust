#![feature(min_generic_const_args)]
#![feature(generic_const_exprs)]
#![expect(incomplete_features)]

struct Both<const is_123: u32 = 3, T> { //~ ERROR generic parameters with a default must be trailing
    a: A<{ B::<1>::M }>, //~ ERROR cannot find type `A` in this scope
    //~| ERROR failed to resolve: use of undeclared type `B`
}

fn main() {}
