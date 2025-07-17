//~ NOTE `align` could refer to a built-in attribute

// Anti-regression test to demonstrate that at least we mitigated breakage from adding a new
// `#[align]` built-in attribute.

// Needs edition >= 2018 macro use behavior.
//@ edition: 2018

macro_rules! align {
    //~^ NOTE `align` could also refer to the macro defined here
    () => {
        /* .. */
    };
}

pub(crate) use align;
//~^ ERROR `align` is ambiguous
//~| NOTE ambiguous name
//~| NOTE ambiguous because of a name conflict with a builtin attribute

fn main() {}
