#![feature(type_alias_impl_trait)]

type PairCoupledTypes: Trait<
    //~^ ERROR: bounds on `type`s in this context have no effect
    //~| ERROR: cannot find trait `Trait` in this scope
    [u32; {
        static FOO: usize; //~ ERROR: free static item without body
    }],
> = impl Trait<
    //~^ ERROR: cannot find trait `Trait` in this scope
    //~| ERROR: unconstrained opaque type
    [u32; {
        static FOO: usize; //~ ERROR: free static item without body
    }],
>;

fn main() {}
