// Regression test for issue #80062 (fixed by `min_const_generics`)

fn sof<T>() -> T { unimplemented!() }

fn test<T>() {
    let _: [u8; sof::<T>()];
    //~^ ERROR generic parameters may not be used in const operations
}

fn main() {}
