//! Never coercions should not justify suggestions that claim to produce an unrelated value.

fn diverging() -> ! {
    panic!()
}

fn call_diverging_function() {
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
    let _: u32 = None::<u32>.unwrap_or(|| -> ! { panic!() });
    //~^ ERROR mismatched types
}

fn map_or_never(value: Option<!>) -> ! {
    value.unwrap_or(&[])
    //~^ ERROR mismatched types
}

fn main() {}
