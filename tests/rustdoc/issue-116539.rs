// Regression test for <https://github.com/rust-lang/rust/issues/116539>.
#![no_std]
#![crate_name = "foo"]

trait IdentifyAccount {
    type A;
}
struct RealSigner {}

impl IdentifyAccount for RealSigner {
    type A = u32;
}

type RealAccountId = <RealSigner as IdentifyAccount>::A;

trait BaseConfig {
    type B;
}

trait Config: BaseConfig<B = RealAccountId> {}

struct GenesisConfig<T: Config> {
    shelves: [<T as BaseConfig>::B; 2],
}
