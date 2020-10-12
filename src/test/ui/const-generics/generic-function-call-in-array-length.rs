// revisions: full min

#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(min, feature(min_const_generics))]

const fn foo(n: usize) -> usize { n * 2 }

fn bar<const N: usize>() -> [u32; foo(N)] {
    //[min]~^ ERROR generic parameters may not be used in const operations
    //[full]~^^ ERROR constant expression depends on a generic parameter
    [0; foo(N)]
    //[min]~^ ERROR generic parameters may not be used in const operations
}

fn main() {}
