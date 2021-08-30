fn ty_param<T>() -> [u8; std::mem::size_of::<T>()] {
    //~^ ERROR generic parameters may not be used in const operations
    todo!()
}

fn const_param<const N: usize>() -> [u8; N + 1] {
    //~^ ERROR generic parameters may not be used in const operations
    todo!()
}

fn main() {}
