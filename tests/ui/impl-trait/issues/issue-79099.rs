struct Bug {
    V1: [(); {
        let f: impl core::future::Future<Output = u8> = async { 1 };
        //~^ ERROR `impl Trait` is not allowed in the type of variable bindings
        //~| ERROR expected identifier
        1
    }],
}

fn main() {}
