//@ check-pass

#[warn(meta_variable_misuse)]
macro_rules! foo {
    ( $($i:ident)* ) => { $($i)+ }; //~ WARN meta-variable repeats with different Kleene operator
}

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
