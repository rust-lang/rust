// Test for #147748, providing additional clarification that default field values aren't compatible
// with type parameters unless going through the type parameter.
#![feature(default_field_values)]
struct Foo<T = String> { //~ NOTE: expected this type parameter
    x: T = String::new(),
    //~^ ERROR: mismatched types
    //~| NOTE: expected type parameter
    //~| NOTE: expected type parameter
    //~| NOTE: the type of default fields referencing type parameters can't be assumed inside the struct defining them
}
fn main() {}
