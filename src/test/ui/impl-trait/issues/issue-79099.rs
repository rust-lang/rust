struct Bug {
    V1: [(); {
        let f: impl core::future::Future<Output = u8> = async { 1 };
        //~^ `impl Trait` isn't allowed within variable binding [E0562]
        //~| expected identifier
        1
    }],
}

fn main() {}
