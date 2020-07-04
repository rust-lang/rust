pub const fn sof<T>() -> usize {
    10
}

fn test<T>() {
    let _: [u8; sof::<T>()];
    //~^ ERROR the size for values of type `T`
}

fn main() {}
