//! Reject candidates justified only by never coercions, but retain useful suggestions for
//! calling diverging functions and using lazy fallbacks.

fn diverging() -> ! {
    panic!()
}

fn call_diverging_function() {
    // Keep the suggestion to call `diverging`: a missing call is still plausible even when
    // the function never returns.
    let _: u32 = diverging;
    //~^ ERROR mismatched types
}

struct Inherent;

impl Inherent {
    fn item() {}
    const ITEM: ! = panic!();
}

fn associated_const() {
    let _: u32 = Inherent::item;
    //~^ ERROR mismatched types
}

fn lazy_fallback() {
    // Keep `unwrap_or_else` as a hint to use a lazy fallback. The explicit `-> !` also needs
    // to be removed for the suggested code to compile; this suggestion does not handle that.
    let _: u32 = None::<u32>.unwrap_or(|| -> ! { panic!() });
    //~^ ERROR mismatched types
}

fn map_or_never(value: Option<!>) -> ! {
    value.unwrap_or(&[])
    //~^ ERROR mismatched types
}

fn main() {}
