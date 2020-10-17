// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

fn ty_param<T>() -> [u8; std::mem::size_of::<T>()] {
    //[full]~^ ERROR constant expression depends on a generic parameter
    //[min]~^^ ERROR generic parameters may not be used in const operations
    todo!()
}

fn const_param<const N: usize>() -> [u8; N + 1] {
    //[full]~^ ERROR constant expression depends on a generic parameter
    //[min]~^^ ERROR generic parameters may not be used in const operations
    todo!()
}

fn main() {}
