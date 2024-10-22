#![feature(const_trait_impl, effects)] //~ WARN the feature `effects` is incomplete

const fn test() -> impl ~const Fn() {
    //~^ ERROR `~const` can only be applied to `#[const_trait]` traits
    //~| ERROR `~const` can only be applied to `#[const_trait]` traits
    const move || { //~ ERROR const closures are experimental
        let sl: &[u8] = b"foo";

        match sl {
            [first, remainder @ ..] => {
                assert_eq!(first, &b'f');
                //~^ ERROR cannot call non-const fn
            }
            [] => panic!(),
        }
    }
}

fn main() {}
