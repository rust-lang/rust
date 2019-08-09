// A few contrived examples where lifetime should (or should not) be parsed as an object type.
// Lifetimes parsed as types are still rejected later by semantic checks.

struct S<'a, T>(&'a u8, T);

fn main() {
    // `'static` is a lifetime argument, `'static +` is a type argument
    let _: S<'static, u8>;
    let _: S<'static, dyn 'static +>;
    //~^ at least one trait is required for an object type
    let _: S<'static, 'static>;
    //~^ ERROR wrong number of lifetime arguments: expected 1, found 2
    //~| ERROR wrong number of type arguments: expected 1, found 0
    let _: S<dyn 'static +, 'static>;
    //~^ ERROR lifetime arguments must be declared prior to type arguments
    //~| ERROR at least one trait is required for an object type
}
