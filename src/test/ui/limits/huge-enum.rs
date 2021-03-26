// build-fail
// normalize-stderr-test "std::option::Option<\[u32; \d+\]>" -> "TYPE"
// normalize-stderr-test "\[u32; \d+\]" -> "TYPE"

// FIXME https://github.com/rust-lang/rust/issues/59774
// normalize-stderr-test "thread.*panicked.*Metadata module not compiled.*\n" -> ""
// normalize-stderr-test "note:.*RUST_BACKTRACE=1.*\n" -> ""

#[cfg(target_pointer_width = "32")]
type BIG = Option<[u32; (1<<29)-1]>;

#[cfg(target_pointer_width = "64")]
type BIG = Option<[u32; (1<<45)-1]>;

fn main() {
    let big: BIG = None;
    //~^ ERROR are too big for the current architecture
}
