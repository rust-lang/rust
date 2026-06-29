// Ensure assoc items work for macro rules.
// Regression test for <https://github.com/rust-lang/rust/issues/156075>.

//@ compile-flags: -Zunstable-options --generate-macro-expansion

#![crate_name = "foo"]

//@ has 'src/foo/assoc-items-macro.rs.html'

macro_rules! first {
    () => {
        type P1 = bool;
        fn u1() {}
    }
}

trait C1 {
    type P1;
    fn u1();
}

impl C1 for u32 {
    //@ matches - '//*[@class="expansion"]/*[@class="expanded"]' 'type P1 = bool;\nfn u1\(\) {}'
    first!();
}
