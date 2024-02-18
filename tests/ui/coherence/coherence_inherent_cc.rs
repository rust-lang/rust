//@ aux-build:coherence_inherent_cc_lib.rs

// Tests that methods that implement a trait cannot be invoked
// unless the trait is imported.

extern crate coherence_inherent_cc_lib;

mod Import {
    // Trait is in scope here:
    use coherence_inherent_cc_lib::TheStruct;
    use coherence_inherent_cc_lib::TheTrait;

    fn call_the_fn(s: &TheStruct) {
        s.the_fn();
    }
}

mod NoImport {
    // Trait is not in scope here:
    use coherence_inherent_cc_lib::TheStruct;

    fn call_the_fn(s: &TheStruct) {
        s.the_fn();
        //~^ ERROR E0599
    }
}

fn main() {}
