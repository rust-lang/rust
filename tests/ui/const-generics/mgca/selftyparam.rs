#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Tr<const N: usize> {
    fn foo() -> [(); const { let _: Self; 1 }];
    //~^ ERROR generic parameters
}

fn main() {}
