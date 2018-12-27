// run-pass
#![crate_type = "lib"]

extern crate my_crate;

pub fn g() {} // (a)

#[macro_export]
macro_rules! unhygienic_macro {
    () => {
        // (1) unhygienic: depends on `my_crate` in the crate root at the invocation site.
        ::my_crate::f();

        // (2) unhygienic: defines `f` at the invocation site (in addition to the above point).
        use my_crate::f;
        f();

        g(); // (3) unhygienic: `g` needs to be in scope at use site.

        $crate::g(); // (4) hygienic: this always resolves to (a)
    }
}

#[allow(unused)]
fn test_unhygienic() {
    unhygienic_macro!();
    f(); // `f` was defined at the use site
}
