// issue #127742
struct Vtable(dyn Cap);
//~^ ERROR missing lifetime specifier

trait Cap<'a> {}

union Transmute {
    t: u64,
    u: &'static Vtable,
}

const G: &'static Vtable = unsafe { Transmute { t: 1 }.u };
fn main() {}
