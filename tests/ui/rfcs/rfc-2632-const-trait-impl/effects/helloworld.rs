// check-pass

// gate-test-effects
// ^ effects doesn't have a gate so we will trick tidy into thinking this is a gate test

#![feature(const_trait_impl, effects, rustc_attrs)]

// ensure we are passing in the correct host effect in always const contexts.

pub const fn hmm</* T, */ #[rustc_host] const host: bool = true>() -> usize {
    if host {
        1
    } else {
        0
    }
}

const _: () = {
    let x = hmm();
    assert!(0 == x);
};

/* FIXME(effects)
pub const fn uwu(x: [u8; hmm::<()>()]) {
    let [] = x;
}
*/

fn main() {}
