#![feature(const_trait_impl, const_closures)]
#![allow(incomplete_features)]

const fn test() -> impl [const] Fn() {
    //~^ ERROR: }: [const] Fn()` is not satisfied
    const move || {
        let sl: &[u8] = b"foo";

        match sl {
            [first, remainder @ ..] => {
                assert_eq!(first, &b'f');
                // FIXME(const_closures) ^ ERROR cannot call non-const function
            }
            [] => panic!(),
        }
    }
}

fn main() {}
