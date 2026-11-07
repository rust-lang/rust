// This test ensures that the macro expansion is correctly handled in cases like:
// `field: !f!`, because the `:` was simply not considered because of how paths
// are handled.

//@ compile-flags: -Zunstable-options --generate-macro-expansion

#![crate_name = "foo"]

//@ has 'src/foo/field-followed-by-exclamation.rs.html'

struct Bar {
    bla: bool,
}

macro_rules! f {
    () => {{ false }}
}

const X: Bar = Bar {
    //@ has - '//*[@class="expansion"]/*[@class="original"]/*[@class="macro"]' 'f!'
    //@ has - '//*[@class="expansion"]/*[@class="original"]' 'f!()'
    //@ has - '//*[@class="expansion"]/*[@class="expanded"]' '{ false }'
    // It includes both original and expanded code.
    //@ has - '//*[@class="expansion"]' '    bla: !{ false }f!()'
    bla: !f!(),
};
