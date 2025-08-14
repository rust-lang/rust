// Regression test for <https://github.com/rust-lang/rust/issues/127545>.
#![crate_type = "lib"]

pub fn foo(arg: Option<&Vec<i32>>) -> Option<&[i32]> {
    arg //~ ERROR 5:5: 5:8: mismatched types [E0308]
}

pub fn bar(arg: Option<&Vec<i32>>) -> &[i32] {
    arg.unwrap_or(&[]) //~ ERROR 9:19: 9:22: mismatched types [E0308]
}

pub fn barzz<'a>(arg: Option<&'a Vec<i32>>, v: &'a [i32]) -> &'a [i32] {
    arg.unwrap_or(v) //~ ERROR 13:19: 13:20: mismatched types [E0308]
}

pub fn convert_result(arg: Result<&Vec<i32>, ()>) -> &[i32] {
    arg.unwrap_or(&[]) //~ ERROR 17:19: 17:22: mismatched types [E0308]
}
