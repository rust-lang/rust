//@ check-pass
//@ compile-flags: -Z span-debug --error-format human
//@ aux-build:test-macros.rs

/*
#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

macro_rules! produce_it {
    ($expr:expr) => {
        #[derive(Print)]
        struct Foo {
            val: [bool; {
                let a = #[cfg_attr(not(FALSE), rustc_dummy(first))] $expr;
                0
            }]
        }
    }
}

produce_it!(#[cfg_attr(not(FALSE), rustc_dummy(second))] {
    #![cfg_attr(not(FALSE), allow(unused))]
    30
});
*/

fn main() {}

/* njn: this test is failing. Getting this output with attributes repeated:

+PRINT-DERIVE INPUT (DISPLAY): struct Foo
+{
+    val :
+    [bool ;
+    {
+        let a = #[rustc_dummy(first)] #[rustc_dummy(second)]
+        #! [allow(unused)] #[rustc_dummy(second)] { #! [allow(unused)] 30 } ;
+        0
+    }]
+}
+

Plus the annoying:

    "error: test compilation failed although it shouldn't!
    failed to decode compiler output as json: line: {"

*/
