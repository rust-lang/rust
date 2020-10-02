// Tests miscellaneous well-formedness examples.
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

pub fn arr_len<const N: usize>() {
    let _: [u8; N + 1];
    //[full]~^ ERROR constant expression depends on a generic parameter
    //[min]~^^ ERROR generic parameters must not be used inside of non trivial
}

struct Const<const N: usize>;

pub fn func_call<const N: usize>() {
    let _: Const::<{N + 1}>;
    //[full]~^ ERROR constant expression depends on a generic parameter
    //[min]~^^ ERROR generic parameters must not be used inside of non trivial
}

fn main() {}
