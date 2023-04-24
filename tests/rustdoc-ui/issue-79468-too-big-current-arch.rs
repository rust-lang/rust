// Regression test for <https://github.com/rust-lang/rust/issues/79468>.
// only-x86_64
// failure-status: 101

const HUGE_SIZE: usize = !0usize / 8;
static MY_TOO_BIG_ARRAY_2: [u8; HUGE_SIZE] = [0x00; HUGE_SIZE];
//~^ ERROR values of the type `[u8; 2305843009213693951]` are too big
