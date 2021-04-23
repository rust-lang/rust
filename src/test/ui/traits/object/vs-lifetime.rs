// A few contrived examples where lifetime should (or should not) be parsed as an object type.
// Lifetimes parsed as types are still rejected later by semantic checks.

struct S<'a, T>(&'a u8, T);

fn main() {
    // `'static` is a lifetime argument, `'static +` is a type argument
    let _: S<'static, u8>;
    let _: S<'static, dyn 'static +>;
    //~^ at least one trait is required for an object type
    let _: S<'static, 'static>;
    //~^ ERROR this struct takes 1 lifetime argument but 2 lifetime arguments were supplied
    //~| ERROR this struct takes 1 type argument but 0 type arguments were supplied
    let _: S<dyn 'static +, 'static>;
    //~^ ERROR type provided when a lifetime was expected
    //~| ERROR at least one trait is required for an object type
}
