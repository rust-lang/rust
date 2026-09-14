// Ensure that C var args (`va_list`) work.
// Regression test for <https://github.com/rust-lang/rust/issues/156486>.

//@ compile-flags: -Zunstable-options --generate-macro-expansion

#![crate_name = "foo"]

//@ has 'src/foo/c-var-args.rs.html'

macro_rules! print {
    () => {
        fn printf(...);
    };
}

//@ has - '//*[@class="expansion"]/*[@class="expanded"]' 'fn printf(...);'
extern "C" {
    print! {}
}
