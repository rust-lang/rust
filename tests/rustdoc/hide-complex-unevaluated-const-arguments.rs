// Test that certain unevaluated constant expression arguments that are
// deemed too verbose or complex and that may leak private or
// `doc(hidden)` struct fields are not displayed in the documentation.
//
// Read the documentation of `rustdoc::clean::utils::print_const_expr`
// for further details.
#![feature(const_trait_impl, generic_const_exprs)]
#![allow(incomplete_features)]

// @has hide_complex_unevaluated_const_arguments/trait.Stage.html
pub trait Stage {
    // A helper constant that prevents const expressions containing it
    // from getting fully evaluated since it doesn't have a body and
    // thus is non-reducible. This allows us to specifically test the
    // pretty-printing of *unevaluated* consts.
    const ABSTRACT: usize;

    // Currently considered "overly complex" by the `generic_const_exprs`
    // feature. If / once this expression kind gets supported, this
    // unevaluated const expression could leak the private struct field.
    //
    // FIXME: Once the line below compiles, make this a test that
    //        ensures that the private field is not printed.
    //
    //const ARRAY0: [u8; Struct { private: () } + Self::ABSTRACT];

    // This assoc. const could leak the private assoc. function `Struct::new`.
    // Ensure that this does not happen.
    //
    // @has - '//*[@id="associatedconstant.ARRAY1"]' \
    //        'const ARRAY1: [u8; { _ }]'
    const ARRAY1: [u8; Struct::new(/* ... */) + Self::ABSTRACT * 1_000];

    // @has - '//*[@id="associatedconstant.VERBOSE"]' \
    //        'const VERBOSE: [u16; { _ }]'
    const VERBOSE: [u16; compute("thing", 9 + 9) * Self::ABSTRACT];

    // Check that we do not leak the private struct field contained within
    // the path. The output could definitely be improved upon
    // (e.g. printing sth. akin to `<Self as Helper<{ _ }>>::OUT`) but
    // right now “safe is safe”.
    //
    // @has - '//*[@id="associatedconstant.PATH"]' \
    //        'const PATH: usize = _'
    const PATH: usize = <Self as Helper<{ Struct { private: () } }>>::OUT;
}

const fn compute(input: &str, extra: usize) -> usize {
    input.len() + extra
}

pub trait Helper<const S: Struct> {
    const OUT: usize;
}

impl<const S: Struct, St: Stage + ?Sized> Helper<S> for St {
    const OUT: usize = St::ABSTRACT;
}

// Currently in rustdoc, const arguments are not evaluated in this position
// and therefore they fall under the realm of `print_const_expr`.
// If rustdoc gets patched to evaluate const arguments, it is fine to replace
// this test as long as one can ensure that private fields are not leaked!
//
// @has hide_complex_unevaluated_const_arguments/trait.Sub.html \
//      '//pre[@class="rust item-decl"]' \
//      'pub trait Sub: Sup<{ _ }, { _ }> { }'
pub trait Sub: Sup<{ 90 * 20 * 4 }, { Struct { private: () } }> {}

pub trait Sup<const N: usize, const S: Struct> {}

pub struct Struct { private: () }

impl Struct {
    const fn new() -> Self { Self { private: () } }
}

impl const std::ops::Add<usize> for Struct {
    type Output = usize;

    fn add(self, _: usize) -> usize { 0 }
}
