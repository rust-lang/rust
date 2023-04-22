//@ build-fail
//@ normalize-stderr-test: "std::option::Option<\[u32; \d+\]>" -> "TYPE"
//@ normalize-stderr-test: "\[u32; \d+\]" -> "TYPE"

//@ compile-flags: -Z mir-opt-level=0
// (The optimizations would remove the values on which this errors.)

#[cfg(target_pointer_width = "32")]
type BIG = Option<[u32; (1<<29)-1]>;

#[cfg(target_pointer_width = "64")]
type BIG = Option<[u32; (1<<45)-1]>;

fn main() {
    let big: BIG = None;
    //~^ ERROR are too big for the current architecture
}
