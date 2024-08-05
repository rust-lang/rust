#![feature(derive_smart_pointer)]

#[pointee]
//~^ ERROR: attribute should be applied to generic type parameters
struct AStruct<
    #[pointee]
    //~^ ERROR: attribute should be applied to generic type parameters
    'lifetime,
    #[pointee]
    //~^ ERROR: attribute should be applied to generic type parameters
    const CONST: usize
> {
    #[pointee]
    //~^ ERROR: attribute should be applied to generic type parameters
    val: &'lifetime ()
}

#[pointee]
//~^ ERROR: attribute should be applied to generic type parameters
enum AnEnum {
    #[pointee]
    //~^ ERROR: attribute should be applied to generic type parameters
    AVariant
}

#[pointee]
//~^ ERROR: attribute should be applied to generic type parameters
mod AModule {}

#[pointee]
//~^ ERROR: attribute should be applied to generic type parameters
fn a_function(
) {}

fn main() {}
