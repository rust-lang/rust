// compile-flags: -Zmiri-check-number-validity

// This test is from https://github.com/rust-lang/miri/issues/1340#issue-600900312.

fn main() {
    let _val = unsafe { std::mem::MaybeUninit::<usize>::uninit().assume_init() };
    //~^ ERROR type validation failed at .value: encountered uninitialized bytes, but expected initialized plain (non-pointer) bytes
}
