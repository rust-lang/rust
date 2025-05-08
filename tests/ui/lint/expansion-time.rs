//@ check-pass

#[warn(meta_variable_misuse)]
macro_rules! foo {
    ( $($i:ident)* ) => { $($i)+ }; //~ WARN meta-variable repeats with different Kleene operator
}

#[warn(missing_fragment_specifier)]
macro_rules! m { ($i) => {} } //~ WARN missing fragment specifier
                              //~| WARN this was previously accepted

#[deprecated = "reason"]
macro_rules! deprecated {
    () => {}
}

#[allow(deprecated)]
mod deprecated {
    deprecated!(); // No warning
}

#[warn(incomplete_include)]
fn main() {
    // WARN see in the stderr file, the warning points to the included file.
    include!("expansion-time-include.rs");
}

//~? WARN include macro expected single expression in source
