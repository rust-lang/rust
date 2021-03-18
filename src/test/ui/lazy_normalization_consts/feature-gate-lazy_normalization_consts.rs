pub const fn sof<T>() -> usize {
    10
}

fn test<T>() {
    let _: [u8; sof::<T>()];
    //~^ ERROR generic parameters may not be used in const operations
}

fn main() {}
