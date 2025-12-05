// Check that `-Zmir-enable-passes=+Inline` does not ICE because of stolen MIR.
//@ test-mir-pass: Inline
// skip-filecheck
#![crate_type = "lib"]

// Randomize `def_path_hash` by defining them under a module with different names
macro_rules! emit {
    ($($m:ident)*) => {$(
        pub mod $m {
            pub fn main() {
                let func = || 123u8;
                func();
            }
        }
    )*};
}

// Increase the chance of triggering the bug
emit!(m00 m01 m02 m03 m04 m05 m06 m07 m08 m09 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19);
