#![feature(const_trait_impl, effects)]

const fn test() -> impl ~const Fn() { //~ ERROR `~const` can only be applied to `#[const_trait]` traits
    //~^ ERROR cycle detected
    const move || { //~ ERROR const closures are experimental
        let sl: &[u8] = b"foo";

        match sl {
            [first, remainder @ ..] => {
                assert_eq!(first, &b'f');
                //~^ ERROR can't compare `&u8` with `&u8`
            }
            [] => panic!(),
        }
    }
}

fn main() {}
