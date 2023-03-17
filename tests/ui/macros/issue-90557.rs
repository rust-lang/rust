struct Example<T, U> {
    foo: T,
    bar: U,
}

macro_rules! impl_example {
    ($($t:ty)+) => {$(
        impl Example<$t> { //~ ERROR struct takes 2 generic arguments but 1 generic argument was supplied
            fn baz() {
                println!(":)");
            }
        }
    )+}
}

impl_example! { u8 }

fn main() {}
