// Regression test for <https://github.com/rust-lang/rust/issues/127545>.
#![crate_type = "lib"]

pub fn foo(arg: Option<&Vec<i32>>) -> Option<&[i32]> {
    arg //~ ERROR 5:5: 5:8: mismatched types [E0308]
}
