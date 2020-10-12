// revisions: full min

#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(min, feature(min_const_generics))]

struct Bug<T> {
    //~^ ERROR parameter `T` is never used
    inner: [(); { [|_: &T| {}; 0].len() }],
    //[min]~^ ERROR generic parameters may not be used in const operations
    //[full]~^^ WARN cannot use constants which depend on generic parameters in types
    //[full]~^^^ WARN this was previously accepted by the compiler
}

fn main() {}
