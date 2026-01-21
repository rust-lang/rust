#![feature(
    unknown_rust_feature,
    //~^ ERROR unknown feature

    // Typo for lang feature
    associated_types_default,
    //~^ ERROR unknown feature
    //~| HELP there is a feature with a similar name

    // Typo for lib feature
    core_intrnisics,
    //~^ ERROR unknown feature
    //~| HELP there is a feature with a similar name
)]

fn main() {}
