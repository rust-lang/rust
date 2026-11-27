// Ensure macro invocations at type position are expanded correctly

//@ compile-flags: -Zunstable-options --generate-macro-expansion

#![crate_name = "foo"]

//@ has 'src/foo/type-macro-expansion.rs.html'

macro_rules! foo {
    () => {
        fn(())
    };
    ($_arg:expr) => {
        [(); 1]
    };
}

fn bar() {
    //@ has - '//*[@class="expansion"]/*[@class="original"]/*[@class="macro"]' 'foo!'
    //@ has - '//*[@class="expansion"]/*[@class="original"]' 'foo!()'
    //@ has - '//*[@class="expansion"]/*[@class="expanded"]' 'fn(())'
    let _: foo!();
    //@ has - '//*[@class="expansion"]/*[@class="original"]/*[@class="macro"]' 'foo!'
    //@ has - '//*[@class="expansion"]/*[@class="original"]' 'foo!(42)'
    //@ has - '//*[@class="expansion"]/*[@class="expanded"]' '[(); 1]'
    let _: foo!(42);
}
