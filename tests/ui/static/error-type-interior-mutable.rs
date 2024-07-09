// A regression test for #124164
// const-eval used to complain because the allocation was mutable (due to the atomic)
// while it expected `{type error}` allocations to be immutable.

static S_COUNT: = std::sync::atomic::AtomicUsize::new(0);
//~^ ERROR missing type for `static` item

fn main() {}
