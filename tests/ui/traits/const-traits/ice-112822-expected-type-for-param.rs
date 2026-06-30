#![feature(const_trait_impl, const_closures, const_cmp)]

const fn test() -> impl [const] Fn() {
    const move || {
        let sl: &[u8] = b"foo";

        match sl {
            [first, remainder @ ..] => {
                assert_eq!(first, &b'f');
                //~^ ERROR is not satisfied
            }
            [] => panic!(),
        }
    }
}

fn main() {}
