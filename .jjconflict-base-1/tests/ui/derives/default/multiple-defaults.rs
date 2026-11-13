//@ compile-flags: --crate-type=lib

// When we get multiple `#[default]` variants, we emit several tool-only suggestions
// to remove all except one of the `#[default]`s.

#[derive(Default)] //~ ERROR multiple declared defaults
enum A {
    #[default] //~ HELP make `B` default
    #[default] //~ HELP make `A` default
    A,
    #[default] // also "HELP make `A` default", but compiletest can't handle multispans
    B,
}

// Originally, we took each defaulted variant and emitted the suggestion for every variant
// with a different identifier, causing an ICE when multiple variants have the same identifier:
// https://github.com/rust-lang/rust/pull/105106
#[derive(Default)] //~ ERROR multiple declared defaults
enum E {
    #[default] //~ HELP make `A` default
    A,
    #[default] //~ HELP make `A` default
    A, //~ ERROR defined multiple times
}

// Then, we took each defaulted variant and emitted the suggestion for every variant
// with a different span, causing an ICE when multiple variants have the same span:
// https://github.com/rust-lang/rust/issues/118119
macro_rules! m {
    { $($id:ident)* } => {
        #[derive(Default)] //~ ERROR multiple declared defaults
        enum F {
            $(
                #[default]
                $id,
            )*
        }
    }
}

m! { A B }
