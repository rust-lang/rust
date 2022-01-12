struct Bug {
    V1: [(); {
        let f: impl core::future::Future<Output = u8> = async { 1 };
        //~^ `impl Trait` not allowed outside of function and inherent method return types
        //~| expected identifier
        1
    }],
}

fn main() {}
