// run-pass
#![feature(rustc_main)]
#![rustc_main(crate::inner::alt_main, naked)]

mod inner {
    fn alt_main(_: isize, _: *const *const u8) -> isize { 0 }
}
