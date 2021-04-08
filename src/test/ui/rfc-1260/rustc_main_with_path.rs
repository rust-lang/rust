// run-pass
#![feature(rustc_main)]
#![rustc_main(crate::inner::alt_main)]

mod inner {
    fn alt_main() {}
}
