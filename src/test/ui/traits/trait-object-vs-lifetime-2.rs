// A few contrived examples where lifetime should (or should not) be parsed as an object type.
// Lifetimes parsed as types are still rejected later by semantic checks.

// compile-flags: -Z continue-parse-after-error

// `'static` is a lifetime, `'static +` is a type, `'a` is a type
fn g() where
    'static: 'static,
    dyn 'static +: 'static + Copy,
    //~^ ERROR at least one non-builtin trait is required for an object type
{}

fn main() {}
