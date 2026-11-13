#![feature(rustc_attrs)]

#[rustc_on_unimplemented(
    on(Self = "{union}", message = "union self type"),
    on(Self = "{enum}", message = "enum self type"),
    on(Self = "{struct}", message = "struct self type"),
    message = "fallback self type `{Self}`"
)]
trait Trait {}

union Union {
    value: u8,
}

enum Enum {
    Variant,
}

struct Struct;

fn needs_trait<T: Trait>() {}

fn main() {
    needs_trait::<Union>();
    //~^ ERROR union self type
    needs_trait::<Enum>();
    //~^ ERROR enum self type
    needs_trait::<Struct>();
    //~^ ERROR struct self type
}
