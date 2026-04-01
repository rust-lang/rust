// Test that we render the deprecated anonymous trait function parameters from Rust 2015 as
// underscores in order not to perpetuate it and for legibility.

//@ edition: 2015
#![expect(anonymous_parameters)]

// Check the "local case" (HIR cleaning) //

//@ has anon_fn_params/trait.Trait.html
pub trait Trait {
    //@ has - '//*[@id="tymethod.required"]' 'fn required(_: Option<i32>, _: impl Fn(&str) -> bool)'
    fn required(Option<i32>, impl Fn(&str) -> bool);
    //@ has - '//*[@id="method.provided"]' 'fn provided(_: [i32; 2])'
    fn provided([i32; 2]) {}
}

// Check the "extern case" (middle cleaning) //

//@ aux-build: ext-anon-fn-params.rs
extern crate ext_anon_fn_params;

//@ has anon_fn_params/trait.ExtTrait.html
//@ has - '//*[@id="tymethod.required"]' 'fn required(_: Option<i32>, _: impl Fn(&str) -> bool)'
//@ has - '//*[@id="method.provided"]' 'fn provided(_: [i32; 2])'
pub use ext_anon_fn_params::Trait as ExtTrait;
