// A few contrived examples where lifetime should (or should not) be parsed as an object type.
// Lifetimes parsed as types are still rejected later by semantic checks.

// compile-flags: -Z continue-parse-after-error

struct S<'a, T>(&'a u8, T);

fn main() {
    // `'static` is a lifetime argument, `'static +` is a type argument
    let _: S<'static, u8>;
    let _: S<'static, 'static +>;
    //~^ at least one non-builtin trait is required for an object type
    let _: S<'static, 'static>;
    //~^ ERROR wrong number of lifetime arguments: expected 1, found 2
    //~| ERROR wrong number of type arguments: expected 1, found 0
    let _: S<'static +, 'static>;
    //~^ ERROR lifetime parameters must be declared prior to type parameters
    //~| ERROR at least one non-builtin trait is required for an object type
}
