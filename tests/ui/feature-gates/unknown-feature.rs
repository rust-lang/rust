#![feature(
    unknown_rust_feature,
    //~^ ERROR unknown feature
    
    // Typo for lang feature
    associated_types_default,
    //~^ ERROR unknown feature

    // Typo for lib feature
    core_intrnisics,
    //~^ ERROR unknown feature
)]

fn main() {}
