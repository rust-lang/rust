// This test is adapted from https://github.com/rust-lang/miri/issues/1340#issue-600900312.
// This test passes because -Zmiri-check-number-validity is not passed.

fn main() {
    let _val1 = unsafe { std::mem::MaybeUninit::<usize>::uninit().assume_init() };
    let _val2 = unsafe { std::mem::MaybeUninit::<i32>::uninit().assume_init() };
    let _val3 = unsafe { std::mem::MaybeUninit::<f32>::uninit().assume_init() };
}
