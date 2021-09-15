// run-pass

#![feature(const_trait_impl)]
#![feature(const_fn_trait_bound)]

const fn answer_p1<F>(f: &F) -> u8
    where
        F: ~const FnOnce() -> u8,
        F: ~const FnMut() -> u8,
        F: ~const Fn() -> u8,
{
    f() * 7
}

const fn three() -> u8 {
    3
}

const fn answer_p2() -> u8 {
    answer_p1(&three)
}

const fn answer<F: ~const Fn() -> u8>(f: &F) -> u8 {
    f() + f()
}

const ANSWER: u8 = answer(&answer_p2);

fn main() {
    assert_eq!(ANSWER, 42)
}
