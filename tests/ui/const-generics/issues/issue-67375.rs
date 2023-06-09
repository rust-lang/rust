// revisions: full min
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(generic_const_exprs))]

struct Bug<T> {
    //[min]~^ ERROR parameter `T` is never used
    inner: [(); { [|_: &T| {}; 0].len() }],
    //[min]~^ ERROR generic parameters may not be used in const operations
    //[full]~^^ ERROR overly complex generic constant
}

fn main() {}
