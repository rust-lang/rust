//@ known-bug: #127742
struct Vtable(dyn Cap);  // missing lifetime

trait Cap<'a> {}

union Transmute {
    t: u64,  // ICEs with u64, u128, or usize. Correctly errors with u32.
    u: &'static Vtable,
}

const G: &'static Vtable = unsafe { Transmute { t: 1 }.u };
